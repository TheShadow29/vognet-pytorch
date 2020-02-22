"""
Loading data correctly
"""
# from extended_config import cfg as conf
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.nn import functional as F
from pathlib import Path
from _init_stuff import Fpath, Arr, ForkedPdb
from yacs.config import CfgNode as CN
import pandas as pd
import h5py
from typing import Tuple
import numpy as np
import json
import copy
from typing import Dict
from munch import Munch
from trn_utils import DataWrap
# import torchtext.vocab as vocab  # use this to load glove vector
from torchtext import vocab
from tqdm import tqdm
import os
import random
from collections import defaultdict
from fairseq.data import Dictionary
import ast
from collections import Counter
import pickle
import copy
from ds4_creator import create_similar_list, create_random_list
from mdl_srl_utils import combine_first_ax
import random
# import warnings
# warnings.filterwarnings('error')

# np.seterr(all='warn')

torch.multiprocessing.set_sharing_strategy('file_system')


class AnetEntDataset(Dataset):
    """
    Dataset class adopted from
    https://github.com/facebookresearch/grounded-video-description
    /blob/master/misc/dataloader_anet.py#L27
    """

    def __init__(self, cfg: CN, ann_file: Fpath, split_type: str = 'train',
                 comm: Dict = None):
        self.cfg = cfg

        # Common stuff that needs to be passed around
        if comm is not None:
            assert isinstance(comm, (dict, Munch))
            self.comm = Munch(comm)
        else:
            self.comm = Munch()

        self.split_type = split_type
        self.ann_file = Path(ann_file)
        assert self.ann_file.suffix == '.csv'
        self.set_args()
        self.load_annotations()

        self.create_glove_stuff()

        h5_proposal_file = h5py.File(
            self.proposal_h5, 'r', driver='core')
        self.num_proposals = h5_proposal_file['dets_num'][:]
        self.label_proposals = h5_proposal_file['dets_labels'][:]

        self.itemgetter = getattr(self, 'simple_item_getter')
        # self.itemgetter = getattr(self, 'gvd_getitem')
        self.test_mode = (split_type != 'test')
        self.after_init()

    def after_init(self):
        pass

    def set_args(self):
        """
        Define the arguments to be used from the cfg
        """
        self.use_gt_prop = self.cfg.ds.use_gt_prop
        # self.vis_attn = False

        # if not self.use_gt_prop:
        # NOTE: These are changed at extended_config/post_proc_config
        self.proposal_h5 = Path(self.cfg.ds.proposal_h5)
        self.feature_root = Path(self.cfg.ds.feature_root)
        # else:
        # self.proposal_h5 = Path(self.cfg.ds.proposal_gt10_h5)
        # self.feature_root = Path(self.cfg.ds.feature_gt10_root)

        # Assert h5 file to read from exists
        assert self.proposal_h5.exists()

        # Assert region features exists
        assert self.feature_root.exists()

        # Assert rgb, motion features exists
        self.seg_feature_root = Path(self.cfg.ds.seg_feature_root)
        assert self.seg_feature_root.exists()

        # Which proposals to be included
        self.prop_thresh = self.cfg.misc.prop_thresh
        self.exclude_bgd_det = self.cfg.misc.exclude_bgd_det

        # Max proposals to be considered
        # By default it is 10 * 100
        self.num_sampled_frm = self.cfg.misc.num_sampled_frm
        self.num_prop_per_frm = self.cfg.misc.num_prop_per_frm

        self.max_proposals = (self.num_sampled_frm *
                              self.num_prop_per_frm)

        self.max_proposal = self.max_proposals
        # Max gt box to consider
        # should consider all, set high
        self.max_gt_box = self.cfg.misc.max_gt_box

        # Assert raw caption file (from activity net captions) exists
        self.raw_caption_file = Path(self.cfg.ds.anet_cap_file)
        assert self.raw_caption_file.exists()

        # Assert act ent caption file with bbox exists
        self.anet_ent_annot_file = Path(self.cfg.ds.anet_ent_annot_file)
        assert self.anet_ent_annot_file.exists()

        # Assert word vocab files exist
        self.dic_anet_file = Path(self.cfg.ds.dic_anet_file)
        assert self.dic_anet_file.exists()

        # temporal attention size
        self.t_attn_size = self.cfg.misc.t_attn_size

        # Sequence length
        self.seq_length = self.cfg.misc.seq_length
        self.seq_per_img = self.cfg.misc.seq_per_img

        # self.att_feat_size = self.cfg.mdl.att_feat_size
        self.num_prop_per_frm = self.cfg.misc.num_prop_per_frm
        self.num_frms = self.cfg.misc.num_sampled_frm
        self.num_props = self.num_prop_per_frm * self.num_frms

    def load_annotations(self):
        """
        Process the annotation file.
        """
        # Load annotation files
        self.annots = pd.read_csv(self.ann_file)

        # Load raw captions
        with open(self.raw_caption_file) as f:
            self.raw_caption = json.load(f)

        # Load anet bbox
        with open(self.anet_ent_annot_file) as f:
            self.anet_ent_captions = json.load(f)

        # Needs to exported as well

        # Load dictionaries
        with open(self.dic_anet_file) as f:
            self.comm.dic_anet = json.load(f)

        # Get detections to index
        self.comm.dtoi = {w: i+1 for w,
                          i in self.comm.dic_anet['wtod'].items()}
        self.comm.itod = {i: w for w, i in self.comm.dtoi.items()}
        self.comm.itow = self.comm.dic_anet['ix_to_word']
        self.comm.wtoi = {w: i for i, w in self.comm.itow.items()}

        self.comm.vocab_size = len(self.comm.itow) + 1
        self.comm.detect_size = len(self.comm.itod)

        # # for convenience
        # for k in self.comm:
        #     val = getattr(self.comm, k)
        #     setattr(self, k, val)

    def create_glove_stuff(self):
        # Load the glove vocab
        self.glove = vocab.GloVe(name='6B', dim=300)

        # get the glove vector for the vg detection cls
        # From Peter's repo
        obj_cls_file = self.cfg.ds.vg_class_file
        # index 0 is the background
        with open(obj_cls_file) as f:
            data = f.readlines()
            classes = ['__background__']
            classes.extend([i.strip() for i in data])

        # for VG classes
        self.comm.vg_cls = classes
        # Extract glove vectors for the Visual Genome Classes
        # TODO: Cleaner implementation possible
        # TODO: Preproc only once
        self.comm.glove_vg_cls = np.zeros((len(classes), 300))
        for i, w in enumerate(classes):
            split_word = w.replace(',', ' ').split(' ')
            vector = []
            for word in split_word:
                if word in self.glove.stoi:
                    vector.append(
                        self.glove.vectors[self.glove.stoi[word]].numpy())
                else:  # use a random vector instead
                    vector.append(2*np.random.rand(300) - 1)

            avg_vector = np.zeros((300))
            for v in vector:
                avg_vector += v

            self.comm.glove_vg_cls[i] = avg_vector/len(vector)

        # category id to labels. +1 becuase 0 is the background label
        # Extract glove vectors for the 431 classes in AnetEntDataset
        # TODO: Cleaner Implementation
        # TODO: Preproc only once
        self.comm.glove_clss = np.zeros((len(self.comm.itod)+1, 300))
        self.comm.glove_clss[0] = 2*np.random.rand(300) - 1  # background
        for i, word in enumerate(self.comm.itod.values()):
            if word in self.glove.stoi:
                vector = self.glove.vectors[self.glove.stoi[word]]
            else:  # use a random vector instead
                vector = 2*np.random.rand(300) - 1
            self.comm.glove_clss[i+1] = vector

        # Extract glove vectors for the words from the vocab
        # TODO: cleaner implementation
        # TODO: preproc only once
        self.glove_w = np.zeros((len(self.comm.wtoi)+1, 300))
        for i, word in enumerate(self.comm.wtoi.keys()):
            vector = np.zeros((300))
            count = 0
            for w in word.split(' '):
                count += 1
                if w in self.glove.stoi:
                    glove_vector = self.glove.vectors[self.glove.stoi[w]]
                    vector += glove_vector.numpy()
                else:  # use a random vector instead
                    random_vector = 2*np.random.rand(300) - 1
                    vector += random_vector
            self.glove_w[i+1] = vector / count

    def __len__(self):
        return len(self.annots)  #
        # return 50

    def __getitem__(self, idx: int):
        return self.itemgetter(idx)

    def pad_stuff(self, props: Arr, mask: Arr = None,
                  max_num: int = None,
                  default_shape: tuple = None,
                  mask_false_val=0) -> Tuple[Arr, Arr]:
        """
        Pads input array to maximum number.
        Optionally takes input a mask, which is also padded
        Outputs the padded array, padded mask

        NOTE: USE WITH CAUTION. Some ambiguity in 1/0 for masking
        """
        assert isinstance(props, (list, np.ndarray, torch.tensor))
        if default_shape is None:
            assert isinstance(props[0], (np.ndarray, torch.tensor))
            default_shape = props[0].shape

        assert default_shape is not None

        prop_len = len(props)
        if max_num is None:
            max_num = self.max_proposals
        if mask is None:
            mask = [mask_false_val] * prop_len

        if not isinstance(props, list):
            props = props.tolist()

        if not isinstance(mask, list):
            mask = mask.tolist()

        if prop_len >= max_num:
            padded_props = props[:max_num]
            padded_mask = mask[:max_num]
        else:
            remainder = max_num - prop_len
            padded_props = props + \
                [np.zeros(default_shape).tolist()] * remainder
            padded_mask = mask + [1] * remainder

        padded_props = np.array(padded_props)
        padded_mask = np.array(padded_mask)

        return padded_props, padded_mask

    def get_props(self, index: int):
        """
        Returns the padded proposals, padded mask, number of proposals
        by reading the h5 files
        """
        # Load num proposals, proposals
        # with h5py.File(self.proposal_h5, 'r',
        #                driver='core') as h5_proposal_file:
        #     num_proposals = int(h5_proposal_file['dets_num'][index])
        #     label_proposals = h5_proposal_file['dets_labels'][index]

        num_proposals = int(self.num_proposals[index])
        label_proposals = self.label_proposals[index]
        proposals = copy.deepcopy(label_proposals[:num_proposals, :])

        # proposal mask to filter out low-confidence proposals or backgrounds
        pnt_mask = (proposals[:, 6] <= self.prop_thresh)
        if self.exclude_bgd_det:
            pnt_mask |= (proposals[:, 5] == 0)

        num_props = min(proposals.shape[0], self.max_proposals)

        padded_props, padded_mask = self.pad_stuff(
            proposals, mask=pnt_mask,
            default_shape=(7,),
            max_num=self.max_proposals,
            mask_false_val=1
        )

        return padded_props, padded_mask, num_props

    def get_features(self, vid_seg_id: str, num_proposals: int, props):
        """
        Returns the region features, rgb-motion features
        """

        vid_id_ix, seg_id_ix = vid_seg_id.split('_segment_')
        seg_id_ix = str(int(seg_id_ix))

        region_feature_file = self.feature_root / f'{vid_seg_id}.npy'
        region_feature = np.load(region_feature_file)
        region_feature = region_feature.reshape(
            -1,
            region_feature.shape[2]
        ).copy()
        assert(num_proposals == region_feature.shape[0])
        if self.cfg.ds.add_prop_to_region:
            region_feature = np.concatenate(
                [region_feature, props[:num_proposals, :5]],
                axis=1
            )

        # load the frame-wise segment feature
        seg_rgb_file = self.seg_feature_root / f'{vid_id_ix[2:]}_resnet.npy'
        seg_motion_file = self.seg_feature_root / f'{vid_id_ix[2:]}_bn.npy'

        if seg_rgb_file.exists() and seg_motion_file.exists():

            seg_rgb_feature = np.load(seg_rgb_file)
            seg_motion_feature = np.load(seg_motion_file)
            seg_feature_raw = np.concatenate(
                (seg_rgb_feature, seg_motion_feature), axis=1)

            return region_feature, seg_feature_raw
        else:
            return -1, -1

    def get_det_word(self, gt_bboxs, caption, bbox_ann):
        # get the present category.
        pcats = []
        for i in range(gt_bboxs.shape[0]):
            pcats.append(gt_bboxs[i, 6])
        # get the orginial form of the caption.
        indicator = []

        # category class, binary class, fine-grain class.
        indicator.append([(0, 0, 0)]*len(caption))
        for i, bbox in enumerate(bbox_ann):
            # if the bbox_idx is not filtered out.
            if bbox['bbox_idx'] in pcats:
                w_idx = bbox['idx']
                ng = bbox['clss']
                bn = (ng != caption[w_idx]) + 1
                fg = bbox['label']
                indicator[0][w_idx] = (
                    self.comm.dtoi[bbox['clss']], bn, fg)

        return indicator

    def get_gt_annots(self, captions, idx: int):
        """
        Create the annotations in the same way as done in
        GVD repo
        Logic is not very clear, but I keep that as future task for the time being
        Only need to take care in evaluation
        """
        assert len(
            captions) == 1,  'Only support one caption per segment for now!'

        # Init bbox ann list
        bbox_ann = []
        bbox_idx = 0
        for caption in captions:
            for i, clss in enumerate(caption['clss']):
                for j, cls in enumerate(clss):
                    # one box might have multiple labels
                    # we don't care about the boxes outside the length limit.
                    # === May need consideration for testing =====
                    # after all our goal is referring, not detection
                    if caption['idx'][i][j] < self.seq_length:
                        bbox_ann.append(
                            {
                                'bbox': caption['bbox'][i],
                                'label': self.comm.dtoi[cls],
                                'clss': cls,
                                'bbox_idx': bbox_idx,
                                'idx': caption['idx'][i][j],
                                'frm_idx': caption['frm_idx'][i]
                            }
                        )
                        bbox_idx += 1

        # (optional) sort the box based on idx
        # CAUTION: DON'T use the following,
        # because SRL arg boxes need the same sequence
        # as the original. Since CLSS is carried over
        # to the SRL annots, we are safe
        # #### bbox_ann = sorted(bbox_ann, key=lambda x: x['idx'])

        # gt_bboxs layout:
        # 0-3: bbox (x1y1x2y2 format)
        # 4: frame idx
        # 5: label (clss basically)
        # 6: bbox_idx (number in the annotation)
        # 7: word index referring to the object
        gt_bboxs = np.zeros((len(bbox_ann), 8))
        for i, bbox in enumerate(bbox_ann):
            gt_bboxs[i, :4] = bbox['bbox']
            gt_bboxs[i, 4] = bbox['frm_idx']
            gt_bboxs[i, 5] = bbox['label']
            gt_bboxs[i, 6] = bbox['bbox_idx']
            gt_bboxs[i, 7] = bbox['idx']

        # Basically, considers only those gt boxes which
        # have area > 0
        if not self.test_mode:  # skip this in test mode
            gt_x = (gt_bboxs[:, 2]-gt_bboxs[:, 0]+1)
            gt_y = (gt_bboxs[:, 3]-gt_bboxs[:, 1]+1)
            gt_area_nonzero = (((gt_x != 1) & (gt_y != 1)))
            gt_bboxs = gt_bboxs[gt_area_nonzero]

        # given the bbox_ann, and caption, this function
        # determine which word belongs to the detection.
        # det_indicator is of shape (len(caption))
        # for each word it produces 3-tuple
        # for those not corresponding to a particular class
        # it is set as (0,0,0)
        # else (category label, binary label, fine-grained label)
        # use of binary label is not clear
        # category label and fine-grained label seem to be the same
        det_indicator = self.get_det_word(
            gt_bboxs, captions[0]['caption'], bbox_ann)
        # fetch the captions
        ncap = len(captions)  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        # convert caption into sequence label.
        # number of captions is usually 1
        cap_seq = np.zeros([ncap, self.seq_length, 5])
        # In case there are multiple captions
        # Right now only one caption
        # Example caption:
        # [{'caption': ['We', 'see', 'a', 'title', 'screen'],
        # 'frm_idx': [0], 'clss': [['screen']],
        # 'bbox': [[0, 0, 719, 403]], 'idx': [[4]]}]
        for i, caption in enumerate(captions):
            j = 0
            while j < len(caption['caption']) and j < self.seq_length:
                is_det = False
                if det_indicator[i][j][0] != 0:
                    cap_seq[i, j, 0] = det_indicator[i][j][0] + \
                        self.comm.vocab_size
                    cap_seq[i, j, 1] = det_indicator[i][j][1]
                    cap_seq[i, j, 2] = det_indicator[i][j][2]
                    cap_seq[i, j, 3] = self.comm.wtoi[caption['caption'][j]]
                    cap_seq[i, j, 4] = self.comm.wtoi[caption['caption'][j]]
                else:
                    cap_seq[i, j, 0] = self.comm.wtoi[caption['caption'][j]]
                    cap_seq[i, j, 4] = self.comm.wtoi[caption['caption'][j]]
                j += 1

        # cap_seq format:
        # if not cls word
        # 0 - word index
        # 1 - 0
        # 2 - 0
        # 3 - 0
        # 4 - word index
        # elif cls word
        # 0 - category class + vocab_size. This is likely category index
        # 1 - binary class (not sure what it means)
        # 2 - fine grained class
        # 3 - word index
        # 4 - word index

        # get the mask of the ground truth bounding box. The data shape is
        # num_caption x num_box x num_seq
        # NOTE: For Training GVD with SRL inputs, this might need some
        # tweaking
        # CAUTION: mask is 0 if true, 1 if false
        box_mask = np.ones((len(captions), gt_bboxs.shape[0], self.seq_length))
        for i in range(gt_bboxs.shape[0]):
            box_mask[0, i, int(gt_bboxs[i][7])] = 0

        gt_bboxs = gt_bboxs[:, :6]

        # get the batch version of the seq and box_mask.
        # basically, if ncap < self.seq_per_img use all
        # else only choose one at random
        if ncap < self.seq_per_img:
            seq_batch = np.zeros([self.seq_per_img, self.seq_length, 4])
            mask_batch = np.zeros(
                [self.seq_per_img, gt_bboxs.shape[0], self.seq_length])
            # we need to subsample (with replacement)
            for q in range(self.seq_per_img):
                ixl = np.random.randint(0, ncap+1)
                seq_batch[q, :] = cap_seq[ixl, :, :4]
                mask_batch[q, :] = box_mask[ixl]
        else:
            ixl = np.random.randint(0, ncap - self.seq_per_img + 1)
            seq_batch = cap_seq[ixl:ixl+self.seq_per_img, :, :4]
            mask_batch = box_mask[ixl:ixl+self.seq_per_img]

        # input_seq is same as seq_batch, but has
        # one extra in seq_length dimension
        # Reason for this is not clear
        input_seq = np.zeros([self.seq_per_img, self.seq_length+1, 4])
        input_seq[:, 1:] = seq_batch

        # gt_seq is only the word indices
        # at most 10 captions at a time
        gt_seq = np.zeros([10, self.seq_length])
        gt_seq[:ncap, :] = cap_seq[:, :, 4]

        padded_gt_bboxs, padded_gt_box_mask = self.pad_stuff(
            gt_bboxs, max_num=self.max_gt_box, default_shape=(6,))

        num_box = min(gt_bboxs.shape[0], self.max_gt_box)

        pad_box_mask = np.ones(
            (self.seq_per_img, self.max_gt_box, self.seq_length+1))
        pad_box_mask[:, :num_box, 1:] = mask_batch[:, :num_box, :]
        out_dict = {
            'seq_len': [
                min(
                    len(caption['caption']),
                    self.seq_length+1
                ) for caption in captions],
            'inp_seq': input_seq,
            'gt_seq': gt_seq,
            'padded_gt_box': padded_gt_bboxs,
            'padded_gt_box_mask': pad_box_mask,
            'num_box': num_box
        }
        return out_dict

    def get_frm_mask(self, proposals, gt_bboxs):
        """
        1 where proposals and gt box don't match
        0 where they match
        We are basically matching the frame indices,
        that is 1 where they belong to different frames
        0 where they belong to same frame.

        In mdl_bbox_utils.py -> bbox_overlaps_batch
        frm_mask ~= frm_mask is used.
        (We have been tricked, we have been backstabbed,
        quite possibly bamboozled)
        """
        # proposals: num_pps
        # gt_bboxs: num_box
        num_pps = proposals.shape[0]
        num_box = gt_bboxs.shape[0]
        return (np.tile(proposals.reshape(-1, 1), (1, num_box)) != np.tile(
            gt_bboxs, (num_pps, 1)))

    def get_seg_feat_for_frms(self, seg_feats, timestamps, duration, idx=None):
        """
        Given seg features of shape num_frms x 3072
        converts to 10 x 3072
        Here 10 is the number of frames used by the mdl
        timestamps contains the start and end time of the clip
        duration is the total length of the video
        note that end-st != dur, since one is for the clip
        other is for the video

        Additionally returns average over the timestamps
        """
        # ctx is the context of the optical flow used
        # 10 means 5 seconds previous, to 5 seconds after
        # This is because optical flow is calculated at
        # 2fps
        ctx = self.cfg.misc.ctx_for_seg_feats
        if timestamps[0] > timestamps[1]:
            # something is wrong in AnetCaptions dataset
            # since only 2 have problems, ignore
            # print(idx, 'why')
            timestamps = timestamps[1], timestamps[0]
        st_time = timestamps[0]
        end_time = timestamps[1]
        duration_clip = end_time - st_time

        num_frms = seg_feats.shape[0]
        frm_ind = np.arange(0, 10)
        frm_time = st_time + (duration_clip / 10) * (frm_ind + 0.5)
        # *2 because of sampling at 2fps
        frm_index_in_seg_feat = np.minimum(np.maximum(
            (frm_time*2).astype(np.int_)-1, 0), num_frms-1)

        st_indices = np.maximum(frm_index_in_seg_feat - ctx - 1, 0)
        end_indices = np.minimum(frm_index_in_seg_feat + ctx + 1, num_frms)

        if not st_indices[0] == end_indices[-1]:
            try:
                seg_feats_frms_glob = seg_feats[st_indices[0]:end_indices[-1]].mean(
                    axis=0)
            except RuntimeWarning:
                import pdb
                pdb.set_trace()
        else:
            print(f'clip duration: {duration_clip}')
            seg_feats_frms_glob = seg_feats[st_indices[0]]

        assert np.all(end_indices - st_indices > 0)
        try:
            if ctx != 0:
                seg_feats_frms = np.vstack([
                    seg_feats[sti:endi, :].mean(axis=0)
                    for sti, endi in zip(st_indices, end_indices)])
            else:
                seg_feats_frms = seg_feats[frm_index_in_seg_feat]
        except RuntimeWarning:
            import pdb
            pdb.set_trace()
            pass
        return seg_feats_frms, seg_feats_frms_glob

    def simple_item_getter(self, idx: int):
        row = self.annots.iloc[idx]

        vid_id = row['vid_id']
        seg_id = str(row['seg_id'])
        vid_seg_id = row['id']
        ix = row['Index']

        # num_segs = self.annots[self.annots.vid_id == vid_id].seg_id.max()
        num_segs = row['num_segs']

        # Get the padded proposals, proposal masks and the number of proposals
        padded_props, pad_pnt_mask, num_props = self.get_props(ix)

        # Get the region features and the segment features
        # Region features are for spatial stuff
        # Segment features are for temporal stuff
        region_feature, seg_feature_raw = self.get_features(
            vid_seg_id, num_proposals=num_props, props=padded_props
        )

        # not accurate, with minor misalignments
        # Get the time stamp information for each segment
        timestamps = self.raw_caption[vid_id]['timestamps'][int(seg_id)]
        # Get the durations for each time stamp
        dur = self.raw_caption[vid_id]['duration']

        # Get the number of frames in the segment
        num_frm = seg_feature_raw.shape[0]

        # basically time stamps
        sample_idx = np.array(
            [
                np.round(num_frm*timestamps[0]*1./dur),
                np.round(num_frm*timestamps[1]*1./dur)
            ]
        )

        sample_idx = np.clip(np.round(sample_idx), 0,
                             self.t_attn_size).astype(int)

        # Get segment features based on the number of frames used
        seg_feature = np.zeros((self.t_attn_size, seg_feature_raw.shape[1]))
        seg_feature[:min(self.t_attn_size, num_frm)
                    ] = seg_feature_raw[:self.t_attn_size]

        seg_feature_for_frms, seg_feature_for_frms_glob = self.get_seg_feat_for_frms(
            seg_feature_raw, timestamps, dur, idx)
        # get gt annotations
        # Get the act ent annotations
        captions = [self.anet_ent_captions[vid_id]['segments'][seg_id]]
        ncap = len(captions)

        gt_annot_dict = self.get_gt_annots(captions, idx)

        pad_gt_bboxs = gt_annot_dict['padded_gt_box']
        num_box = gt_annot_dict['num_box']

        frm_mask = self.get_frm_mask(
            padded_props[:num_props, 4], pad_gt_bboxs[:num_box, 4])
        pad_frm_mask = np.ones((self.max_proposals, self.max_gt_box))
        pad_frm_mask[:num_props, :num_box] = frm_mask

        # 0 - number of captions
        # 1 - number of proposals
        # 2 - number of boxes
        # 3 - segment id
        # 4 - number of segments
        # 5 - start time stamp
        # 6 - end time stamp
        num = torch.FloatTensor(
            [ncap, num_props, num_box, int(seg_id),
             # max(self.num_seg_per_vid[vid_id]
             # )+1,
             num_segs,
             timestamps[0]*1./dur,
             timestamps[1]*1./dur]
        )  # 3 + 4 (seg_id, num_of_seg_in_video, seg_start_time, seg_end_time)

        pad_pnt_mask = torch.tensor(pad_pnt_mask).long()
        pnt_mask2 = torch.cat(
            (pad_pnt_mask.new(1).fill_(0),
             pad_pnt_mask),
            dim=0
        )

        pad_region_feature = np.zeros(
            (self.max_proposals, region_feature.shape[1]))
        pad_region_feature[:num_props] = region_feature[:num_props]

        out_dict = {
            'seg_feature': torch.from_numpy(seg_feature).float(),
            'seg_feature_for_frms': torch.from_numpy(
                seg_feature_for_frms).float(),
            'seg_feature_for_frms_glob': torch.from_numpy(
                seg_feature_for_frms_glob).float(),
            'inp_seq': torch.from_numpy(gt_annot_dict['inp_seq']).long(),
            'gt_seq': torch.from_numpy(gt_annot_dict['gt_seq']).long(),
            'num': num,
            'seq_len': torch.tensor(gt_annot_dict['seq_len']).long(),
            'num_props': torch.tensor(num_props).long(),
            'num_box': torch.tensor(num_box).long(),
            'pad_proposals': torch.tensor(padded_props).float(),
            'pad_gt_bboxs': torch.tensor(pad_gt_bboxs).float(),
            'pad_gt_box_mask': torch.tensor(
                gt_annot_dict['padded_gt_box_mask']).byte(),
            'seg_id': torch.tensor(int(seg_id)).long(),
            'idx': torch.tensor(idx).long(),
            'ann_idx': torch.tensor(idx).long(),
            'pad_region_feature': torch.tensor(pad_region_feature).float(),
            'pad_frm_mask': torch.tensor(pad_frm_mask).byte(),
            'sample_idx': torch.tensor(sample_idx).long(),
            'pad_pnt_mask': pad_pnt_mask.byte(),
            'pad_pnt_mask2': pnt_mask2.byte(),
        }

        return out_dict


class AnetVerbDataset(AnetEntDataset):
    def fix_via_ast(self, df):
        for k in df.columns:
            first_word = df.iloc[0][k]
            if isinstance(first_word, str) and (first_word[0] in '[{'):
                df[k] = df[k].apply(
                    lambda x: ast.literal_eval(x))
        return df

    def after_init(self):
        if self.split_type == 'train':
            self.srl_annots = pd.read_csv(self.cfg.ds.trn_verb_ent_file)
        elif self.split_type == 'valid':
            self.srl_annots = pd.read_csv(self.cfg.ds.val_verb_ent_file)

        elif self.split_type == 'test':
            self.srl_annots = pd.read_csv(self.cfg.ds.val_verb_ent_file)

        else:
            raise NotImplementedError

        assert hasattr(self, 'srl_annots')
        self.srl_annots = self.fix_via_ast(self.srl_annots)

        with open(self.cfg.ds.arg_vocab_file, 'rb') as f:
            self.arg_vocab = pickle.load(f)

        self.srl_arg_len = self.cfg.misc.srl_arg_length
        self.box_per_srl_arg = self.cfg.misc.box_per_srl_arg

        if self.cfg.ds.proc_type == 'one_verb':
            self.itemgetter = getattr(self, 'verb_item_getter')
            self.max_srl_in_sent = 1
        elif self.cfg.ds.proc_type == 'one_sent':
            self.itemgetter = getattr(self, 'sent_item_getter')
            self.srl_annots = list(self.srl_annots.groupby(by=['ann_ind']))
            self.max_srl_in_sent = self.cfg.misc.max_srl_in_sent

    def __len__(self):
        # if self.split_type == 'train':
        # return 50
        return len(self.srl_annots)
        # return 50

    def get_det_word_simple(self, caption, srl_px_to_ix_map, srl_row):

        indicator = []
        proc_idxs2 = srl_row.process_idx2

        proc_clss2 = srl_row.process_clss
        # category class, binary class, fine-grain class.
        indicator.append([(0, 0, 0)]*len(caption))
        for pind, pidx in enumerate(proc_idxs2):
            for pind2, pidx2 in enumerate(pidx):
                if not isinstance(pidx2, list):  #
                    pidxs3 = [pidx2]
                else:
                    pidxs3 = pidx2
                for pidx3 in pidxs3:
                    if pidx3 in srl_px_to_ix_map:
                        px2 = srl_px_to_ix_map[pidx3]
                        indicator[0][px2] = (
                            self.comm.dtoi[proc_clss2[pind][pind2]], 0, 0)

        return indicator

    def pad_words_with_vocab(self, out_list, vocab=None, pad_len=-1, defm=[1]):
        if pad_len == -1:
            return out_list
        else:
            curr_len = len(out_list)
            if curr_len > pad_len:
                return out_list[:pad_len]
            else:
                if vocab is not None and hasattr(vocab, 'itos'):
                    assert vocab.itos[1] == '<pad>'
                out_list += defm * (pad_len - curr_len)
                return out_list

    def pidx2list(self, pidx):
        """
        Converts process_idx2 to single list
        """
        lst = []
        for p1 in pidx:
            if not isinstance(p1, list):
                p1 = [p1]
            for p2 in p1:
                if not isinstance(p2, list):
                    p2 = [p2]
                for p3 in p2:
                    assert not isinstance(p3, list)
                    lst.append(p3)
        return lst

    def get_srl_anns(self, srl_row, out=None):
        """
        To output dictionary of whatever srl needs
        1. tags
        2. args with st, end ixs
        3. box ind matching
        """
        srl_row = copy.deepcopy(srl_row)

        def word_to_int_vocab(words, vocab, pad_len=-1):
            """
            A convenience function to convert words
            into their indices given a vocab.
            Using Anet Vocab only.
            Optionally, pad answers as well
            """
            out_list = []
            if hasattr(vocab, 'stoi'):
                vocs = vocab.stoi
            else:
                vocs = vocab
            for w in words:
                if w in vocs:
                    out_list.append(int(vocs[w]))
                else:
                    if hasattr(vocab, 'UNK'):
                        unk_word = vocab.UNK
                    else:
                        unk_word = 'UNK'
                    out_list.append(int(vocs[unk_word]))
            curr_len = len(out_list)
            return self.pad_words_with_vocab(out_list,
                                             vocab, pad_len=pad_len), curr_len

        vis_set = ['ARG0', 'ARG1', 'ARG2', 'ARGM-LOC']
        # want to get the arguments and the word indices
        srl_args, srl_words_inds = [list(t) for t in zip(*srl_row.req_pat_ix)]
        srl_args_visual_msk = self.pad_words_with_vocab(
            [s in vis_set for s in srl_args],
            pad_len=self.srl_arg_len, defm=[0])
        # get the words from their indices
        srl_arg_words = [[srl_row.words[ix]
                          for ix in y] for y in srl_words_inds]

        # Tags are converted via tag vocab
        # seq_len, len
        tag_seq = [srl_row.tags[ix] for y in srl_words_inds for ix in y]
        tag_word_ind, _ = word_to_int_vocab(
            # srl_row.tags,
            tag_seq,
            self.arg_vocab['arg_tag_vocab'],
            pad_len=self.seq_length
        )

        # Argument Names (ARG0/V/) are  converted to indices
        # Max num of arguments is kept to be self.srl_arg_len

        assert 'V' in srl_args
        verb_ind_in_srl = srl_args.index('V')
        if not verb_ind_in_srl <= self.srl_arg_len - 1:
            verb_ind_in_srl = 0

        srl_arg_inds, srl_arg_len = word_to_int_vocab(
            srl_args, self.arg_vocab['arg_vocab'],
            pad_len=self.srl_arg_len
        )
        if srl_arg_len > self.srl_arg_len:
            srl_arg_len = self.srl_arg_len
        # defm: is the default matrix to be used
        defm = tuple([[1] * self.seq_length, 0])
        # convert the words to their indices using the vocab
        # for every argument
        srl_arg_words_ind_length = self.pad_words_with_vocab(
            [word_to_int_vocab(
                srl_arg_w, self.comm.wtoi, pad_len=self.seq_length) for
                srl_arg_w in srl_arg_words],
            pad_len=self.srl_arg_len, defm=[defm])

        # Unzip to get the word indices and their lengths for
        # each argument separately
        srl_arg_words_ind, srl_arg_words_length = [
            list(t) for t in zip(*srl_arg_words_ind_length)]

        # This is used to convert
        # [[ARG0: w1,w2], [ARG1: w5,..]] ->
        # [w1,w2,w5]
        # Basically, convert
        # [0] 0,1 -> 0,1
        # [1] 0,1 -> 40, 41
        # and so on
        # Finally, use this with index_select
        srl_arg_word_list = [
            torch.arange(0+st, 0+st+wlen)
            for st, wlen in zip(
                range(
                    0,
                    self.seq_length*self.srl_arg_len,
                    self.seq_length), srl_arg_words_length)
        ]

        # Concat above list
        srl_arg_words_list = torch.cat(srl_arg_word_list, dim=0).tolist()
        # Create the mask to be used with index select
        srl_arg_words_mask = self.pad_words_with_vocab(
            srl_arg_words_list, pad_len=self.seq_length, defm=[-1]
        )

        # Get the start and end positions
        # these are used to retrieve
        # LSTM outputs of the sentence
        # to the argument vectors
        srl_arg_word_list_tmp = [
            0] + torch.cumsum(
                torch.tensor(srl_arg_words_length),
                dim=0).tolist()
        srl_arg_words_capture = [
            (min(x, self.seq_length-1), min(y-1, self.seq_length-1))
            if wlen > 0 else (0, 0)
            for x, y, wlen in zip(
                srl_arg_word_list_tmp[:-1],
                srl_arg_word_list_tmp[1:],
                srl_arg_words_length
            )
        ]

        # This is used to retrieve in argument form from
        # the sentence form
        # Basically, [w1,w2,w5] -> [[ARG0: w1,w2], [ARG1: w5]]
        # Restrict to max len because scatter is used later
        srl_arg_words_map_inv = [
            y_ix for y_ix, y in enumerate(
                srl_words_inds[:self.srl_arg_len]) for ix in y]

        # Also, pad it
        srl_arg_words_map_inv = self.pad_words_with_vocab(
            srl_arg_words_map_inv,
            pad_len=self.seq_length,
            defm=[0]
        )

        # The following creates a binary mask for the sequence_length
        # [1] * seq_cnt for every ARG row
        # This is applied to the scatter output
        defm = [[0] * self.seq_length]
        seq_cnt = sum(srl_arg_words_length)
        srl_arg_words_binary_mask = self.pad_words_with_vocab(
            [self.pad_words_with_vocab(
                [1]*seq_cnt, pad_len=self.seq_length, defm=[0])
             for srl_arg_w in srl_arg_words],
            pad_len=self.srl_arg_len, defm=defm)

        # Get the set of visual words
        vis_idxs_set = set(self.pidx2list(srl_row.process_idx2))
        # Create a map for getting which are the visual words
        srl_arg_words_conc_ix = [ix for y in srl_words_inds for ix in y]
        # Create the binary mask
        srl_vis_words_binary_mask = self.pad_words_with_vocab(
            [1 if srl_vw1 in vis_idxs_set else 0
             for srl_vw1 in srl_arg_words_conc_ix],
            pad_len=self.seq_length, defm=[0])

        # The following are used to map the gt boxes
        # The first is the srl argument, followed by an
        # indicator wheather the box is valid or not
        # third is if valid which boxes to look at
        srl_args, srl_arg_box_indicator, srl_arg_box_inds = [
            list(t) for t in zip(*srl_row.req_cls_pats_mask)
        ]

        # srl boxes, and their lengths are stored in a list
        srl_boxes = []
        srl_boxes_lens = []
        for s1_ind, s1 in enumerate(srl_arg_box_inds):
            mult = min(
                len(s1),
                self.box_per_srl_arg
            ) if srl_arg_box_indicator[s1_ind] == 1 else 0

            s11 = [x if x_ix < self.box_per_srl_arg else 0 for x_ix,
                   x in enumerate(s1)]
            srl_boxes.append(self.pad_words_with_vocab(
                s11, pad_len=self.box_per_srl_arg, defm=[0]))
            srl_boxes_lens.append(self.pad_words_with_vocab(
                [1]*mult, pad_len=self.box_per_srl_arg, defm=[0]))

        # They are then padded
        srl_boxes = self.pad_words_with_vocab(
            srl_boxes,
            pad_len=self.srl_arg_len,
            defm=[[0]*self.box_per_srl_arg]
        )
        srl_boxes_lens = self.pad_words_with_vocab(
            srl_boxes_lens,
            pad_len=self.srl_arg_len,
            defm=[[0]*self.box_per_srl_arg]
        )

        # An indicator wheather the boxes are valid
        srl_arg_boxes_indicator = self.pad_words_with_vocab(
            srl_arg_box_indicator, pad_len=self.srl_arg_len, defm=[0])

        # if not torch.all(torch.tensor(srl_boxes).long() < self.box_per_srl_arg):
        #     import pdb
        #     pdb.set_trace()
        #     pass
        out_dict = {
            # Tags are indexed (B-V -> 4)
            'srl_tag_word_ind': torch.tensor(tag_word_ind).long(),
            # Tag word len available elsewhere, hence removed
            # 'tag_word_len': torch.tensor(tag_word_len).long(),
            # 1 if arg is in ARG1-2/LOC else 0
            'srl_args_visual_msk': torch.tensor(srl_args_visual_msk).long(),
            # ARGs are indexed (ARG0 -> 4, V -> 2)
            'srl_arg_inds': torch.tensor(srl_arg_inds).long(),
            # How many args are considered (ARG0, V,ARG1, ARGM), would be 4
            'srl_arg_len': torch.tensor(srl_arg_len).long(),
            # the above but in mask format
            'srl_arg_inds_msk': torch.tensor(
                [1] * srl_arg_len + [0]*(self.srl_arg_len - srl_arg_len)
            ).long(),
            # Where the verb is located, in prev eg, it would be 1
            'verb_ind_in_srl': torch.tensor(verb_ind_in_srl).long(),
            # num_srl_args x seq_len: for each srl_arg, what is the seq
            # so ARG0: The woman -> [[1946, 4307, ...],...]
            'srl_arg_words_ind': torch.tensor(srl_arg_words_ind).long(),
            # The corresponding lengths of each num_srl
            'srl_arg_words_length': torch.tensor(srl_arg_words_length).long(),
            # num_srl_args x seq_len, 1s upto the seq_len of the whole
            # srl_sent: This is used in scatter operation
            'srl_arg_words_binary_mask': torch.tensor(
                srl_arg_words_binary_mask).long(),
            # Similar to previous, but 1s only at places
            # which are visual words. Used for scatter + GVD
            'srl_vis_words_binary_mask': torch.tensor(
                srl_vis_words_binary_mask).long(),
            # seq_len, but contains in the indices to be gathered
            # from num_srl_args x seq_len -> num_srl_args*seq_len
            # via index_select
            'srl_arg_word_mask': torch.tensor(srl_arg_words_mask).long(),
            # seq_len basically
            'srl_arg_word_mask_len': torch.tensor(min(sum(
                srl_arg_words_length), self.seq_length)).long(),
            # containing start and end points of the words to be collected
            'srl_arg_words_capture': torch.tensor(srl_arg_words_capture).long(),
            # used scatter + GVD
            'srl_arg_words_map_inv': torch.tensor(srl_arg_words_map_inv).long(),
            # box indices in gt boxes
            'srl_boxes': torch.tensor(srl_boxes).long(),
            # mask on which of them to choose
            'srl_boxes_lens': torch.tensor(srl_boxes_lens).long(),
            'srl_arg_boxes_mask': torch.tensor(srl_arg_boxes_indicator).long()
        }
        return out_dict

    def collate_dict_list(self, dict_list, pad_len=None):
        out_dict = {}
        keys = list(dict_list[0].keys())
        num_dl = len(dict_list)
        if pad_len is None:
            pad_len = self.max_srl_in_sent
        for k in keys:
            dl_list = [dl[k] for dl in dict_list]
            dl_list_pad = self.pad_words_with_vocab(
                dl_list,
                pad_len=pad_len, defm=[dl_list[0]])
            out_dict[k] = torch.stack(dl_list_pad)
        return out_dict, num_dl

    def sent_item_getter(self, idx):
        """
        get vidseg at a time, multiple verbs
        """

        ann_ind, srl_rows = self.srl_annots[idx]
        out = self.simple_item_getter(ann_ind)
        out_dict_list = [self.get_srl_anns(srl_rows.iloc[ix], out)
                         for ix in range(len(srl_rows))]
        srl_row_indices = self.pad_words_with_vocab(
            srl_rows.index.tolist(),
            pad_len=self.max_srl_in_sent)
        out_dict, num_verbs = self.collate_dict_list(out_dict_list)
        out_dict['num_verbs'] = torch.tensor(num_verbs).long()
        out_dict['ann_idx'] = torch.tensor(ann_ind).long()
        out_dict['sent_idx'] = torch.tensor(idx).long()
        out_dict['srl_verb_idxs'] = torch.tensor(srl_row_indices).long()
        out.update(out_dict)
        return out

    def get_for_one_verb(self, srl_row, idx, out=None):
        out_dict_list = [self.get_srl_anns(srl_row, out)]
        out_dict, num_verbs = self.collate_dict_list(out_dict_list)
        out_dict['num_verbs'] = torch.tensor(num_verbs).long()
        out_dict['ann_idx'] = torch.tensor(srl_row.ann_ind).long()
        out_dict['sent_idx'] = torch.tensor(idx).long()
        out_dict['srl_verb_idxs'] = torch.tensor([idx]).long()
        return out_dict

    def verb_item_getter(self, idx):
        """
        get verb items, one at a time
        """
        srl_row = self.srl_annots.loc[idx]
        out = self.simple_item_getter(srl_row.ann_ind)
        out_dict = self.get_for_one_verb(srl_row, idx, out)
        # out_dict['srl_verb_idxs'] = torch.tensor(srl_row_indices).long()
        out.update(out_dict)
        return out


class AnetVerbDataset_GVD(AnetVerbDataset):
    def recompute_gt_seq_inp_seq(self, srl_words, srl_px_to_ix_map, srl_row):

        # det_indicator = self.get_det_word(
        # gt_bboxs, srl_words, bbox_ann)
        det_indicator = self.get_det_word_simple(
            srl_words, srl_px_to_ix_map, srl_row)

        i = 0
        j = 0
        srl_words2 = [
            srw if srw in self.comm.wtoi else 'UNK' for srw in srl_words]
        caption = {'caption': srl_words2}
        cap_seq = np.zeros([1, self.seq_length, 5])
        while j < len(caption['caption']) and j < self.seq_length:
            if det_indicator[i][j][0] != 0:
                cap_seq[i, j, 0] = det_indicator[i][j][0] + \
                    self.comm.vocab_size
                cap_seq[i, j, 1] = det_indicator[i][j][1]
                cap_seq[i, j, 2] = det_indicator[i][j][2]

                cap_seq[i, j, 3] = self.comm.wtoi[caption['caption'][j]]
                cap_seq[i, j, 4] = self.comm.wtoi[caption['caption'][j]]
            else:
                cap_seq[i, j, 0] = self.comm.wtoi[caption['caption'][j]]
                cap_seq[i, j, 4] = self.comm.wtoi[caption['caption'][j]]
            j += 1

        seq_batch = np.zeros([1, self.seq_length, 4])
        seq_batch[0, :] = cap_seq[0, :, :4]

        input_seq = np.zeros([self.seq_length+1, 4])
        input_seq[1:] = seq_batch[0, :]

        gt_seq = np.zeros([self.seq_length])
        gt_seq[:] = cap_seq[0, :, 4]

        seq_len = min(self.seq_length+1, len(srl_words2))
        return {
            'inp_seq': torch.tensor(input_seq).long(),
            'gt_seq': torch.tensor(gt_seq).long(),
            'seq_len': torch.tensor(seq_len).long()
        }

    def get_new_seq_for_gvd(self, srl_row, out=None):
        srl_args, srl_words_inds = [list(t) for t in zip(*srl_row.req_pat_ix)]
        srl_arg_words_conc = [srl_row.words[ix]
                              for y in srl_words_inds for ix in y]
        srl_arg_words_conc_ix = [ix for y in srl_words_inds for ix in y]
        srl_arg_words_conc_pix_to_ix_map = {
            ix: ix_ind for ix_ind, ix in enumerate(srl_arg_words_conc_ix)}

        # Recompute input_seq, gt_seq
        out_dict1 = self.recompute_gt_seq_inp_seq(
            srl_arg_words_conc,
            srl_arg_words_conc_pix_to_ix_map,
            srl_row)
        return out_dict1

    def get_srl_anns(self, srl_row, out=None):
        out_dict = super().get_srl_anns(srl_row, out)
        out_dict1 = self.get_new_seq_for_gvd(srl_row, out)
        out_dict.update(out_dict1)
        return out_dict


class AVDS4:
    def __len__(self):
        # if self.split_type == 'train':
        # return 50
        return len(self.srl_annots)
        # return 50

    def after_init(self):
        if self.split_type == 'train':
            srl_annot_file = self.cfg.ds.trn_ds4_inds
            arg_dict_file = self.cfg.ds.trn_ds4_dicts
        elif self.split_type == 'valid':
            srl_annot_file = self.cfg.ds.val_ds4_inds
            arg_dict_file = self.cfg.ds.val_ds4_dicts
        elif self.split_type == 'test':
            srl_annot_file = self.cfg.ds.val_ds4_inds
            arg_dict_file = self.cfg.ds.val_ds4_dicts
        else:
            raise NotImplementedError

        self.srl_annots = pd.read_csv(srl_annot_file)
        assert hasattr(self, 'srl_annots')

        self.srl_annots = self.fix_via_ast(self.srl_annots)
        # self.srl_annots = self.srl_annots.loc[[762, 13005]]
        # self.srl_annots = self.srl_annots.loc[[13098, 460]]
        # self.srl_annots = self.srl_annots.loc[[959, 14646]]
        # THIS MAY BE A BUG
        # import pdb
        # pdb.set_trace()
        # self.srl_annots = self.srl_annots[self.srl_annots.ds4_msk != -1]
        with open(arg_dict_file) as f:
            self.arg_dicts = json.load(f)

        assert self.cfg.ds.proc_type == 'one_verb'
        self.max_srl_in_sent = 1

        if self.split_type == 'train':
            self.ds4_sample = self.cfg.ds.trn_ds4_sample
            assert self.ds4_sample in set(['ds4', 'random', 'ds4_random'])
        elif self.split_type == 'valid' or self.split_type == 'test':
            self.ds4_sample = self.cfg.ds.val_ds4_sample
            assert self.ds4_sample in set(['ds4', 'random'])
        else:
            raise NotImplementedError

        if self.ds4_sample == 'random':
            self.more_idx_collector = getattr(self, 'get_random_more_idx')
        elif self.ds4_sample == 'ds4':
            self.more_idx_collector = getattr(self, 'get_more_idxs')
        elif self.ds4_sample == 'ds4_random':
            self.more_idx_collector = getattr(
                self, 'get_ds4_and_random_more_idx')
        else:
            raise NotImplementedError

        if self.split_type == 'train':
            ds4_sigm_num_inp = self.cfg.misc.trn_ds4_sigm_num_inp
        elif self.split_type in set(['valid', 'test']):
            ds4_sigm_num_inp = self.cfg.misc.val_ds4_sigm_num_inp
        else:
            raise NotImplementedError
        if self.cfg.ds.ds4_type == 'sigmoid':
            self.ds4_inp_len = ds4_sigm_num_inp
            self.itemcollector = getattr(self, 'verb_item_getter_ds4_sigmoid')
        elif self.cfg.ds.ds4_type == 'sigmoid_single_q':
            self.ds4_inp_len = ds4_sigm_num_inp
            self.itemcollector = getattr(
                self, 'verb_item_getter_ds4_sigmoid_single'
            )
        elif self.cfg.ds.ds4_type == 'single':
            self.ds4_inp_len = 1
            self.itemcollector = getattr(
                self, 'verb_item_getter_ds4_sigmoid_single'
            )

            # self.itemcollector = getattr(self, 'verb_item_getter_ds4_sigmoid')
        else:
            raise NotImplementedError

        if self.cfg.ds.ds4_screen == 'screen_spatial':
            self.itemgetter = getattr(
                self, 'verb_item_getter_ds4_screen_spatial')
            self.append_everywhere = False
        elif self.cfg.ds.ds4_screen == 'screen_temporal':
            self.itemgetter = getattr(
                self, 'verb_item_getter_ds4_screen_temporal')
            self.append_everywhere = False
        elif self.cfg.ds.ds4_screen == 'screen_sep':
            self.itemgetter = getattr(
                self, 'verb_item_getter_screen_sep')
            self.append_everywhere = True
            # self.append_everywhere = False
        else:
            raise NotImplementedError

        # Whether to shuffle among the four screens
        self.ds4_shuffle = self.cfg.ds.shuffle_ds4

        with open(self.cfg.ds.arg_vocab_file, 'rb') as f:
            self.arg_vocab = pickle.load(f)

        self.srl_arg_len = self.cfg.misc.srl_arg_length
        self.box_per_srl_arg = self.cfg.misc.box_per_srl_arg

    def get_ds4_and_random_more_idx(self, idx):
        """
        Either choose at random or
        choose ds4
        """
        if np.random.random() < 0.5:
            return self.get_random_more_idx(idx)
        else:
            return self.get_more_idxs(idx)

    def get_random_more_idx(self, idx):
        """
        Returns set of random ds4 idxs
        """
        if self.split_type == 'train':
            more_idxs, _ = create_random_list(self.cfg,
                                              self.srl_annots, idx)
        elif self.split_type == 'valid' or self.split_type == 'test':
            # obtain predefined idxs
            more_idxs = self.srl_annots.RandDS4_Inds.loc[idx]

        if len(more_idxs) > self.ds4_inp_len - 1:
            more_idxs_new_keys = np.random.choice(
                list(more_idxs.keys()),
                min(len(more_idxs), self.ds4_inp_len-1),
                replace=False
            )
            more_idxs = {k: more_idxs[k] for k in more_idxs_new_keys}
        return more_idxs

    def get_more_idxs(self, idx):
        """
        Returns the set of ds4 idxs to consider
        """
        if self.split_type == 'train':
            more_idxs, _ = create_similar_list(self.cfg, self.arg_dicts,
                                               self.srl_annots, idx)
        elif self.split_type == 'valid' or self.split_type == 'test':
            # obtain predefined idxs
            more_idxs = self.srl_annots.DS4_Inds.loc[idx]

        if len(more_idxs) > self.ds4_inp_len - 1:
            more_idxs_new_keys = np.random.choice(
                list(more_idxs.keys()),
                min(len(more_idxs), self.ds4_inp_len-1),
                replace=False
            )
            more_idxs = {k: more_idxs[k] for k in more_idxs_new_keys}

        return more_idxs

    def verb_item_getter(self, idx):
        """
        get verb items, one at a time
        """
        srl_row = self.srl_annots.loc[idx]
        out = self.simple_item_getter(srl_row.ann_ind)
        out_dict = self.get_srl_anns(srl_row, out)
        out_dict['ann_idx'] = torch.tensor(srl_row.ann_ind).long()
        out_dict['sent_idx'] = torch.tensor(idx).long()
        out.update(out_dict)
        return out

    def verb_item_getter_ds4_screen_spatial(self, idx):
        """
        Use DS4 Indices.
        The output is such that we have
        four screens which being played at
        the same time. The goal is to choose
        the correct screen, and ground the
        correct object in that screen.

        Currently, we implement as
        - 4 x region features
        - 4 x temporal features
        However, only one of the four videos
        has the correct answer.

        The way the model sees the input is
        like one single video with 4 screens.
        This is to get away from the problem of
        no prediction score for the whole
        video is generated by the model.

        An alternative is to have scores
        using BCE loss which makes the boxes independent.
        This function implements the former.
        See verb_item_getter_ds4_sigmoid for the latter
        """
        def reshuffle_boxes(inp_t):
            n = inp_t.size(0)
            inp_t = inp_t.view(
                n, self.num_frms, self.num_prop_per_frm, *inp_t.shape[2:]
            ).transpose(0, 1).contiguous().view(
                self.num_frms * n * self.num_prop_per_frm, *inp_t.shape[2:]
            )
            return inp_t

        def process_props(
                props, shift=720,
                keepdim=False, reshuffle_box=False
        ):
            """
            props: n x 1000 x 7
            NOTE: may need to resize
            """
            n, num_props, pdim = props.shape
            delta = torch.arange(n) * shift
            delta = delta.view(n, 1, 1).expand(
                n, num_props, pdim)
            delta_msk = props.new_zeros(*props.shape)
            delta_msk[..., [0, 2]] = 1
            delta = delta.float() * delta_msk.float()

            props_new = props + delta
            if reshuffle_box:
                props_new = reshuffle_boxes(props_new)

            if keepdim:
                return props_new.view(n, num_props, pdim)
            return props_new.view(n*num_props, pdim)

        def process_gt_boxs(gt_boxs, nums):
            gt_box1 = [gtb for gt_box, n1 in zip(
                gt_boxs, nums) for gtb in gt_box[:n1]]
            if len(gt_box1) == 0:
                gt_box1 = [gt_boxs[0, 0]]
            try:
                gt_box1 = torch.stack(gt_box1)
                out = F.pad(
                    gt_box1, (0, 0, 0, self.max_gt_box - len(gt_box1)),
                    mode='constant', value=0
                )

                assert out.shape == (self.max_gt_box, gt_boxs.size(2))
            except:
                ForkedPdb().set_trace()

            return out

        def process_gt_boxs_msk(gt_boxs, nums):
            gt_box1 = [
                gtb for gt_box, n1 in zip(gt_boxs, nums)
                for gtb in gt_box[0, :n1]]
            if len(gt_box1) == 0:
                gt_box1 = [gt_boxs[0, 0]]
            gt_box1 = torch.stack(gt_box1)
            out = F.pad(
                gt_box1, (0, 0, 0, self.max_gt_box - len(gt_box1)),
                mode='constant', value=0
            )

            return out.unsqueeze(0)

        out_dict = self.itemcollector(idx)
        # assert not self.cfg.ds.shuffle_ds4
        num_cmp = len(out_dict['new_srl_idxs'])

        # need to make all the proposals
        # x axis + n delta
        # note that final num_cmp = 1 for videos
        # for lang side B x num_verbs where
        # num_verbs = 4
        # num_cmp1 = out_dict['pad_proposals'].size(0)
        out_dict['num_props'] = out_dict['num_props'].sum(dim=-1)
        # out_dict['num_box'] = out_dict['num_box'].sum(dim=-1)
        out_dict['num_cmp'] = torch.tensor(1)
        out_dict['pad_proposals'] = process_props(
            out_dict['pad_proposals'], keepdim=False, reshuffle_box=True
        )

        num_box = out_dict['num_box'].sum(dim=-1)

        out_dict['pad_gt_bboxs'] = process_gt_boxs(
            process_props(
                out_dict['pad_gt_bboxs'], keepdim=True
            ),
            out_dict['num_box']
        )
        out_dict['pad_gt_box_mask'] = process_gt_boxs_msk(
            out_dict['pad_gt_box_mask'], out_dict['num_box']
        )

        tcmp = out_dict['target_cmp'].item()
        nboxes = [0] + out_dict['num_box'].cumsum(dim=0).tolist()
        new_pos = nboxes[tcmp]
        x1 = out_dict['srl_boxes']
        x2 = out_dict['srl_boxes_lens']
        x1[x2 > 0] += new_pos

        out_dict['num_box2'] = out_dict['num_box'].clone()

        # out_dict['num_box2'] = out_dict['num_box']
        out_dict['num_box'] = num_box

        frm_mask = self.get_frm_mask(
            out_dict['pad_proposals'][:, 4],
            out_dict['pad_gt_bboxs'][:num_box, 4]
        )

        pad_frm_mask = np.ones((num_cmp * self.max_proposals, self.max_gt_box))
        pad_frm_mask[:, :num_box] = frm_mask
        out_dict['pad_frm_mask'] = torch.from_numpy(pad_frm_mask).byte()

        out_dict['pad_pnt_mask'] = reshuffle_boxes(
            out_dict['pad_pnt_mask']
        )
        out_dict['pad_pnt_mask2'] = reshuffle_boxes(
            out_dict['pad_pnt_mask2'][:, 1:].contiguous()
        )

        out_dict['pad_region_feature'] = reshuffle_boxes(
            out_dict['pad_region_feature']
        )

        out_dict['seg_feature'] = combine_first_ax(
            out_dict['seg_feature'], keepdim=False)
        out_dict['seg_feature_for_frms'] = combine_first_ax(
            out_dict['seg_feature_for_frms'].transpose(0, 1).contiguous(),
            keepdim=False
        )
        out_dict['sample_idx'] = combine_first_ax(
            out_dict['sample_idx'], keepdim=False
        )

        return out_dict

    def verb_item_getter_ds4_screen_temporal(self, idx):
        """
        Similar to spatial, but stack in the temporal
        dimension.
        """
        # For temporal stacking: do the following:
        # mostly everything is stacked temporally,
        # only the durations would perhaps change?
        # bboxes frame ids would change
        # Caveats: Videos are not of equal length
        # The above is also applicable to spatial
        def process_props(props, shift=10, keepdim=False,
                          reshuffle_box=False):
            """
            props: n x 1000 x 7
            NOTE: may need to resize
            """
            n, num_props, pdim = props.shape
            delta = torch.arange(n) * shift
            delta = delta.view(n, 1, 1).expand(n, num_props, pdim)
            delta_msk = props.new_zeros(*props.shape)
            delta_msk[..., [4]] = 1
            delta = delta.float() * delta_msk.float()

            props_new = props + delta
            # if reshuffle_box:
            #     props_new = props_new.view(
            #         n, self.num_frms, self.num_prop_per_frm, pdim
            #     ).transpose(0, 1).contiguous()

            if keepdim:
                return props_new.view(n, num_props, pdim)
            return props_new.view(n*num_props, pdim)

        def process_gt_boxs(gt_boxs, nums):
            gt_box1 = [
                gtb for gt_box, n1 in zip(gt_boxs, nums)
                for gtb in gt_box[:n1]]
            if len(gt_box1) == 0:
                gt_box1 = [gt_boxs[0, 0]]
            gt_box1 = torch.stack(gt_box1)
            return F.pad(
                gt_box1, (0, 0, 0, self.max_gt_box - len(gt_box1)),
                mode='constant', value=0
            )

        def process_gt_boxs_msk(gt_boxs, nums):
            gt_box1 = [
                gtb for gt_box, n1 in zip(gt_boxs, nums)
                for gtb in gt_box[0, :n1]]
            if len(gt_box1) == 0:
                gt_box1 = [gt_boxs[0, 0]]
            gt_box1 = torch.stack(gt_box1)
            out = F.pad(
                gt_box1, (0, 0, 0, self.max_gt_box - len(gt_box1)),
                mode='constant', value=0
            )

            return out.unsqueeze(0)

        # Stack proposals in frames

        out_dict = self.itemcollector(idx)
        # assert not self.cfg.ds.shuffle_ds4
        num_cmp = len(out_dict['new_srl_idxs'])
        # need to make all the proposals
        # x axis + n delta
        # note that final num_cmp = 1 for videos
        # for lang side B x num_verbs where
        # num_verbs = 4
        # num_cmp1 = out_dict['pad_proposals'].size(0)
        num_props = out_dict['num_props'].sum(dim=-1)
        out_dict['num_props'] = num_props
        num_box = out_dict['num_box'].sum(dim=-1)
        out_dict['num_cmp'] = torch.tensor(1)

        out_dict['pad_proposals'] = process_props(
            out_dict['pad_proposals'], keepdim=False)
        # re-do gt boxes

        out_dict['pad_gt_bboxs'] = process_gt_boxs(process_props(
            out_dict['pad_gt_bboxs'], keepdim=True), out_dict['num_box'])

        out_dict['pad_gt_box_mask'] = process_gt_boxs_msk(
            out_dict['pad_gt_box_mask'], out_dict['num_box']
        )

        tcmp = out_dict['target_cmp'].item()
        nboxes = [0] + out_dict['num_box'].cumsum(dim=0).tolist()
        new_pos = nboxes[tcmp]
        x1 = out_dict['srl_boxes']
        x2 = out_dict['srl_boxes_lens']
        x1[x2 > 0] += new_pos

        out_dict['num_box2'] = out_dict['num_box'].clone()
        out_dict['num_box'] = num_box
        # frame mask is tricky, so redo
        # out_dict['pad_frm_mask'] = combine_first_ax(
        # out_dict['pad_frm_mask'], keepdim=False)

        frm_mask = self.get_frm_mask(
            out_dict['pad_proposals'][:, 4],
            out_dict['pad_gt_bboxs'][:num_box, 4])
        pad_frm_mask = np.ones((num_cmp * self.max_proposals, self.max_gt_box))
        pad_frm_mask[:, :num_box] = frm_mask

        out_dict['pad_region_feature'] = combine_first_ax(
            out_dict['pad_region_feature'], keepdim=False
        )

        out_dict['pad_frm_mask'] = torch.from_numpy(pad_frm_mask).byte()
        out_dict['pad_pnt_mask'] = combine_first_ax(
            out_dict['pad_pnt_mask'], keepdim=False)
        out_dict['pad_pnt_mask2'] = combine_first_ax(
            out_dict['pad_pnt_mask2'][:, 1:].contiguous(), keepdim=False)
        out_dict['seg_feature'] = combine_first_ax(
            out_dict['seg_feature'], keepdim=False)
        out_dict['seg_feature_for_frms'] = combine_first_ax(
            out_dict['seg_feature_for_frms'], keepdim=False)
        out_dict['sample_idx'] = combine_first_ax(
            out_dict['sample_idx'], keepdim=False
        )
        out_dict['num'] = combine_first_ax(
            out_dict['num'], keepdim=False)

        # out_dict['sample_idx_mask'] = com

        return out_dict

    def verb_item_getter_screen_sep(self, idx):
        """
        When we want separate videos
        """
        return self.itemcollector(idx)

    def verb_item_getter_ds4_sigmoid_single(self, idx):
        """
        Pass the gt query only
        """
        def append_to_every_dict(dct_list, new_dct):
            "append a dict to every dict in a list of dicts"
            for dct in dct_list:
                dct.update(new_dct)
            return

        def shuffle_list_from_perm(lst, perm):
            return [lst[ix] for ix in perm]
        # more_idxs = self.get_more_idxs(idx)
        more_idxs = self.more_idx_collector(idx)
        # if self.cfg.ds.shuffle_ds4:

        new_idxs = [idx]
        lemma_verbs = self.srl_annots.lemma_verb
        curr_verb = lemma_verbs.loc[idx]
        verb_cmp = [1]
        verb_list = [curr_verb]

        if self.split_type == 'train':
            cons = 0
            while len(new_idxs) < self.ds4_inp_len:
                for arg_name, arg_ids in more_idxs.items():
                    if len(new_idxs) < self.ds4_inp_len:
                        arg_id_to_append = arg_ids[cons]
                        # TODO: should be removable
                        if arg_id_to_append != -1:
                            new_idxs += [arg_id_to_append]
                            new_verb = lemma_verbs.loc[arg_id_to_append]
                            verb_cmp += [int(new_verb == curr_verb)]
                            verb_list += [new_verb]
                cons += 1
        else:
            cons = 0
            for arg_name, arg_ids in more_idxs.items():
                if len(new_idxs) < self.ds4_inp_len:
                    arg_id_to_append = arg_ids[cons]
                    # TODO: should be removable
                    if arg_id_to_append != -1:
                        new_idxs += [arg_id_to_append]
                        new_verb = lemma_verbs.loc[arg_id_to_append]
                        verb_cmp += [int(new_verb == curr_verb)]
                        verb_list += [new_verb]

        if self.cfg.ds.shuffle_ds4:
            simple_permute = torch.randperm(len(new_idxs))
        else:
            simple_permute = torch.arange(len(new_idxs))

        simple_permute_inv = simple_permute.argsort()
        simple_permute = simple_permute.tolist()
        simple_permute_inv = simple_permute_inv.tolist()
        targ_cmp = simple_permute_inv[0]

        new_idxs = shuffle_list_from_perm(new_idxs, simple_permute)
        verb_cmp = shuffle_list_from_perm(verb_cmp, simple_permute)
        verb_list = shuffle_list_from_perm(verb_list, simple_permute)

        ann_id_list = [self.srl_annots.loc[ix].ann_ind for ix in new_idxs]
        new_out_dicts = [self.simple_item_getter(ann_ix) for ann_ix
                         in ann_id_list]

        srl_row = self.srl_annots.loc[idx]
        out_dict_verb_for_idx = self.get_srl_anns(srl_row, new_out_dicts[0])

        # out_dict_verb_for_idx['ann_idx'] = torch.tensor(srl_row.ann_ind).long()
        # Append to every dict
        if self.append_everywhere:
            append_to_every_dict(new_out_dicts, out_dict_verb_for_idx)
            collated_out_dicts, num_cmp = self.collate_dict_list(
                new_out_dicts, pad_len=self.ds4_inp_len
            )
        else:
            collated_out_dicts, num_cmp = self.collate_dict_list(
                new_out_dicts, pad_len=self.ds4_inp_len)
            out_dict_verb_for_idx_coll, _ = self.collate_dict_list(
                [out_dict_verb_for_idx], pad_len=1)
            collated_out_dicts.update(out_dict_verb_for_idx_coll)

        new_srl_idxs_pad = self.pad_words_with_vocab(
            new_idxs, pad_len=self.ds4_inp_len)

        # verb_cross_cmp = np.ones((num_cmp, num_cmp))
        verb_cmp_pad = self.pad_words_with_vocab(
            verb_cmp, pad_len=self.ds4_inp_len, defm=[0])

        if len(verb_list) > self.ds4_inp_len:
            verb_list = verb_list[:self.ds4_inp_len]

        verb_list_np = np.array(verb_list)
        verb_cross_cmp = verb_list_np[:, None] == verb_list_np
        verb_cross_cmp_msk = np.ones(verb_cross_cmp.shape)

        verb_cross_cmp = np.pad(
            verb_cross_cmp,
            (0, self.ds4_inp_len - len(verb_list)),
            mode='constant', constant_values=0
        )

        verb_cross_cmp_msk = np.pad(
            verb_cross_cmp_msk,
            (0, self.ds4_inp_len - len(verb_list)),
            mode='constant', constant_values=0
        )

        num_cmp_arr = np.pad(
            np.eye(num_cmp, num_cmp),
            (0, self.ds4_inp_len - num_cmp),
            mode='constant', constant_values=0
        )

        sp_pad = [ix for ix in range(
            num_cmp, num_cmp + self.ds4_inp_len-num_cmp)]
        simple_permute = simple_permute + sp_pad
        assert len(simple_permute) == self.ds4_inp_len
        simple_permute_inv = simple_permute_inv + sp_pad

        out_dict_verb = {}
        out_dict_verb['permute'] = torch.tensor(simple_permute).long()
        out_dict_verb['permute_inv'] = torch.tensor(
            simple_permute_inv).long()
        out_dict_verb['target_cmp'] = torch.tensor(targ_cmp).long()
        out_dict_verb['new_srl_idxs'] = torch.tensor(
            new_srl_idxs_pad).long()
        out_dict_verb['sent_idx'] = torch.tensor(idx).long()
        out_dict_verb['num_cmp'] = torch.tensor(num_cmp).long()
        out_dict_verb['num_cmp_msk'] = torch.tensor(
            [1]*num_cmp + [0] * (self.ds4_inp_len - num_cmp))
        out_dict_verb['num_cross_cmp_msk'] = torch.from_numpy(num_cmp_arr)
        out_dict_verb['verb_cmp'] = torch.tensor(verb_cmp_pad).long()
        out_dict_verb['verb_cross_cmp'] = torch.from_numpy(
            verb_cross_cmp).long()
        out_dict_verb['verb_cross_cmp_msk'] = torch.from_numpy(
            verb_cross_cmp_msk).long()

        collated_out_dicts.update(out_dict_verb)

        return collated_out_dicts

    def verb_item_getter_ds4_sigmoid(self, idx):
        """
        See `verb_item_getter_ds4_screen` for full details.
        This implements the latter.
        Note that here, the order is irrelevant,
        the model treats each of them independently.
        """
        # out_dict = self.verb_item_getter(idx)
        # srl_row = self.srl_annots.loc[idx]

        # more_idxs = self.get_more_idxs(idx)
        more_idxs = self.more_idx_collector(idx)

        new_idxs = [idx]
        lemma_verbs = self.srl_annots.lemma_verb
        curr_verb = lemma_verbs.loc[idx]
        verb_cmp = [1]
        verb_list = [curr_verb]

        for arg_name, arg_ids in more_idxs.items():
            if len(new_idxs) < self.ds4_inp_len:
                arg_id_to_append = arg_ids[0]
                if arg_id_to_append != -1:
                    new_idxs += [arg_id_to_append]
                    new_verb = lemma_verbs.loc[arg_id_to_append]
                    verb_cmp += [int(new_verb == curr_verb)]
                    verb_list += [new_verb]

        new_out_dicts = [self.verb_item_getter(
            new_idx) for new_idx in new_idxs]

        collated_out_dicts, num_cmp = self.collate_dict_list(
            new_out_dicts, pad_len=self.ds4_inp_len)

        new_srl_idxs_pad = self.pad_words_with_vocab(
            new_idxs, pad_len=self.ds4_inp_len)

        verb_cmp_pad = self.pad_words_with_vocab(
            verb_cmp, pad_len=self.ds4_inp_len, defm=[2])

        # verb_list_pad = self.pad_words_with_vocab(
        # verb_list, pad_len=self.ds4_inp_len, defm=[-1])
        if len(verb_list) > self.ds4_inp_len:
            verb_list = verb_list[:self.ds4_inp_len]

        verb_list_np = np.array(verb_list)
        verb_cross_cmp = verb_list_np[:, None] == verb_list_np
        verb_cross_cmp_msk = np.ones(verb_cross_cmp.shape)

        verb_cross_cmp = np.pad(
            verb_cross_cmp,
            (0, self.ds4_inp_len - len(verb_list)),
            mode='constant', constant_values=0
        )

        verb_cross_cmp_msk = np.pad(
            verb_cross_cmp_msk,
            (0, self.ds4_inp_len - len(verb_list)),
            mode='constant', constant_values=0
        )

        # verb_cmp_pad_np = np.array(verb_cmp_pad)
        # verb_cross_cmp_msk1d = verb_cmp_pad != 2
        # verb_cross_cmp_msk2d = (
        #     verb_cross_cmp_msk1d[:, None] == verb_cross_cmp_msk1d)

        num_cmp_arr = np.pad(
            np.eye(num_cmp, num_cmp),
            (0, self.ds4_inp_len - num_cmp),
            mode='constant', constant_values=0
        )

        out_dict_verb = {}
        out_dict_verb['new_srl_idxs'] = torch.tensor(
            new_srl_idxs_pad).long()
        out_dict_verb['sent_idx'] = torch.tensor(idx).long()
        out_dict_verb['num_cmp'] = torch.tensor(num_cmp).long()
        out_dict_verb['num_cmp_msk'] = torch.tensor(
            [1]*num_cmp + [0] * (self.ds4_inp_len - num_cmp))
        out_dict_verb['verb_cmp'] = torch.tensor(verb_cmp_pad).long()
        out_dict_verb['num_cross_cmp_msk'] = torch.from_numpy(num_cmp_arr)
        out_dict_verb['verb_cross_cmp'] = torch.from_numpy(
            verb_cross_cmp).long()
        out_dict_verb['verb_cross_cmp_msk'] = torch.from_numpy(
            verb_cross_cmp_msk).long()

        collated_out_dicts.update(out_dict_verb)
        return collated_out_dicts


class AnetVerbDS4(AVDS4, AnetVerbDataset):
    def __init__(self, cfg: CN, ann_file: Fpath, split_type: str = 'train',
                 comm: Dict = None):
        AnetVerbDataset.__init__(self, cfg, ann_file, split_type, comm)
        # AVDS4.after_init(self)


class AnetVerbDS4Eval_GVD(AVDS4, AnetVerbDataset_GVD):
    def __init__(self, cfg: CN, ann_file: Fpath, split_type: str = 'train',
                 comm: Dict = None):
        AnetVerbDataset_GVD.__init__(self, cfg, ann_file, split_type, comm)
        # AVDS4.after_init(self)


class BatchCollator:
    """
    Need to redefine this perhaps
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.after_init()

    def after_init(self):
        pass

    def __call__(self, batch):
        out_dict = {}

        # nothing needs to be done
        all_keys = list(batch[0].keys())
        batch_size = len(batch)
        for k in all_keys:
            shape = batch[0][k].shape
            if not all([b[k].shape == shape for b in batch]):
                ForkedPdb().set_trace()
            out_dict[k] = torch.stack(
                    [b[k] for b in batch])
        assert all([len(v) == batch_size for k, v in out_dict.items()])

        # num_cmp = out_dict['new_srl_idxs'].size(1)
        # simple_permute = torch.arange(num_cmp)
        # simple_permute_inv = simple_permute.argsort()
        # out_dict['permute_inv'] = torch.stack(
        #     [simple_permute_inv for b in batch])
        # out_dict['permute'] = torch.stack(
        #     [simple_permute for b in batch])
        # out_dict['target_cmp'] = torch.tensor([0]*batch_size).long()

        return out_dict


class BatchCollatorDS4(BatchCollator):
    """
    BatchCollator for DS4
    """

    def after_init(self):
        self.perm_invariant_set = ['sent_idx', 'num_cmp']

        self.double_perm_set = set(
            ['verb_cross_cmp', 'verb_cross_cmp_msk', 'num_cross_cmp_msk']
        )

        self.perm_invariant_set = set(self.perm_invariant_set)

    def __call__(self, batch):
        out_dict = {}

        # nothing needs to be done
        all_keys = list(batch[0].keys())

        batch_size = len(batch)
        num_cmp = len(batch[0][all_keys[0]])
        if self.cfg.ds.shuffle_ds4:
            simple_permute = torch.randperm(num_cmp)
            simple_permute_inv = simple_permute.argsort()
            out_dict['permute_inv'] = torch.stack(
                [simple_permute_inv for b in batch])
            out_dict['permute'] = torch.stack(
                [simple_permute for b in batch])
            out_dict['target_cmp'] = torch.tensor(
                [simple_permute_inv[0].item()]*batch_size).long()
        else:
            simple_permute = torch.arange(num_cmp)
            simple_permute_inv = simple_permute.argsort()
            out_dict['permute_inv'] = torch.stack(
                [simple_permute_inv for b in batch])
            out_dict['permute'] = torch.stack(
                [simple_permute for b in batch])
            out_dict['target_cmp'] = torch.tensor([0]*batch_size).long()
        for k in all_keys:
            shape = batch[0][k].shape
            assert all([b[k].shape == shape for b in batch])
            if k not in self.perm_invariant_set:
                assert shape[0] == num_cmp
                stack_list = [b[k][simple_permute] for b in batch]
                if k in self.double_perm_set:
                    stack_list = [s[:, simple_permute] for s in stack_list]
                out_dict[k] = torch.stack(stack_list)
            else:
                out_dict[k] = torch.stack(
                    [b[k] for b in batch])
        return out_dict


class NewDistributedSampler(DistributedSampler):
    """
    Same as default distributed sampler of pytorch
    Just has another argument for shuffle
    Allows distributed in validation/testing as well
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


def make_data_sampler(dataset: Dataset, shuffle: bool,
                      distributed: bool) -> Sampler:
    if distributed:
        return NewDistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def get_dataloader(cfg, dataset: Dataset, is_train: bool) -> DataLoader:
    is_distributed = cfg.do_dist
    images_per_gpu = cfg.train.bs if is_train else cfg.train.bsv
    nw = cfg.train.nw if is_train else cfg.train.nwv
    if is_distributed:
        # DistributedDataParallel
        batch_size = images_per_gpu
        num_workers = nw
    elif cfg.do_dp:
        # DataParallel
        batch_size = images_per_gpu * cfg.num_gpus
        num_workers = nw * cfg.num_gpus
    else:
        batch_size = images_per_gpu
        num_workers = nw

    if is_train:
        shuffle = True
    else:
        shuffle = False if not is_distributed else True
        # shuffle = False

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    # if ((cfg.ds.ds4_type == 'sigmoid' or cfg.ds.ds4_type == 'sigmoid_single_q')
    #         and cfg.ds.ds4_screen == 'screen_sep'):
    #     collator = BatchCollatorDS4(cfg)
    # else:
    collator = BatchCollator(cfg)

    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, drop_last=is_train,
                      num_workers=num_workers,
                      collate_fn=collator)


def get_data(cfg):
    # Get which dataset to use
    # ds_name = cfg.ds_to_use
    if cfg.ds.proc_type == 'one_sent':
        DS = AnetEntDataset
    elif cfg.ds.proc_type == 'one_verb':
        if not cfg.ds.do_ds4:
            if cfg.mdl.name == 'gvd':
                DS = AnetVerbDataset_GVD
            else:
                DS = AnetVerbDataset
        else:
            if cfg.mdl.name == 'gvd':
                DS = AnetVerbDS4Eval_GVD
            else:
                DS = AnetVerbDS4
    else:
        raise NotImplementedError
    # Training file
    trn_ann_file = cfg.ds['trn_ann_file']
    trn_ds = DS(cfg=cfg, ann_file=trn_ann_file,
                split_type='train')
    trn_dl = get_dataloader(cfg, trn_ds, is_train=True)

    # Validation file
    val_ann_file = cfg.ds['val_ann_file']
    val_ds = DS(cfg=cfg, ann_file=val_ann_file,
                split_type='valid')
    val_dl = get_dataloader(cfg, val_ds, is_train=False)

    test_ann_file = cfg.ds['test_ann_file']
    test_ds = DS(cfg=cfg, ann_file=test_ann_file,
                 split_type='test')
    test_dl = get_dataloader(cfg, test_ds, is_train=False)

    data = DataWrap(path=cfg.misc.tmp_path, train_dl=trn_dl, valid_dl=val_dl,
                    test_dl={'test0': test_dl})
    return data


if __name__ == '__main__':
    from extended_config import cfg as conf
    cfg = conf
    # cfg.train.nw = 0
    cfg.train.bs = 50
    cfg.train.nw = 0
    cfg.ds.do_ds4 = True

    cfg.ds.ds4_screen = 'screen_spatial'
    # cfg.ds.ds4_screen = 'screen_sep'
    # cfg.ds.ds4_type = 'sigmoid_single_q'
    cfg.ds.ds4_type = 'single'
    # trn_ds = AnetEntDataset(
    #     cfg, ann_file=cfg.ds.trn_ann_file, split_type='train')

    data = get_data(cfg)

    diter = iter(data.train_dl)
    batch = next(diter)
    # for _ in tqdm(range(len(data.train_dl))):
    #     try:
    #         batch = next(diter)
    #     except FileNotFoundError:
    #         print('why')
    #         continue
    # # out = trn_ds[16151]
    # batch = next(iter(data.train_dl))
    # batch = next(iter(data.valid_dl))
