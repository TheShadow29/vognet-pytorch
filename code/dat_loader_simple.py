"""
Simplified Data Loading
"""
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.nn import functional as F
from pathlib import Path
from _init_stuff import Fpath, Arr, yaml
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
from torchtext import vocab
import ast
import pickle
from ds4_creator import create_similar_list, create_random_list
from mdl_srl_utils import combine_first_ax

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

        # self.create_glove_stuff()

        h5_proposal_file = h5py.File(
            self.proposal_h5, 'r', driver='core')
        self.num_proposals = h5_proposal_file['dets_num'][:]
        self.label_proposals = h5_proposal_file['dets_labels'][:]

        self.itemgetter = getattr(self, 'simple_item_getter')
        self.test_mode = (split_type != 'test')
        self.after_init()

    def after_init(self):
        pass

    def set_args(self):
        """
        Define the arguments to be used from the cfg
        """
        # self.use_gt_prop = self.cfg.ds.use_gt_prop

        # NOTE: These are changed at extended_config/post_proc_config
        dct = self.cfg.ds[f'{self.cfg.ds.exp_setting}']
        self.proposal_h5 = Path(dct['proposal_h5'])
        self.feature_root = Path(dct['feature_root'])

        # Max proposals to be considered
        # By default it is 10 * 100
        self.num_frms = self.cfg.ds.num_sampled_frm
        self.num_prop_per_frm = dct['num_prop_per_frm']
        self.max_proposals = self.num_prop_per_frm * self.num_frms

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

        # Assert raw caption file (from activity net captions) exists
        self.raw_caption_file = Path(self.cfg.ds.anet_cap_file)
        assert self.raw_caption_file.exists()

        # Assert act ent caption file with bbox exists
        self.anet_ent_annot_file = Path(self.cfg.ds.anet_ent_annot_file)
        assert self.anet_ent_annot_file.exists()

        # Assert word vocab files exist
        self.dic_anet_file = Path(self.cfg.ds.anet_ent_split_file)
        assert self.dic_anet_file.exists()

        # Max gt box to consider
        # should consider all, set high
        self.max_gt_box = self.cfg.ds.max_gt_box

        # temporal attention size
        self.t_attn_size = self.cfg.ds.t_attn_size

        # Sequence length
        self.seq_length = self.cfg.ds.max_seq_length

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

    def __len__(self):
        return len(self.annots)  #
        # return 50

    def __getitem__(self, idx: int):
        return self.itemgetter(idx)

    def pad_words_with_vocab(
            self, out_list,
            voc=None, pad_len=-1, defm=[1]):
        """
        Input is a list.
        If curr_len < pad_len: pad remaining with default value
        Instead, if cur_len > pad_len: trim the input
        """
        curr_len = len(out_list)
        if pad_len == -1 or curr_len == pad_len:
            return out_list
        else:
            if curr_len > pad_len:
                return out_list[:pad_len]
            else:
                if voc is not None and hasattr(voc, 'itos'):
                    assert voc.itos[1] == '<pad>'
                out_list += defm * (pad_len - curr_len)
                return out_list

    def get_props(self, index: int):
        """
        Returns the padded proposals, padded mask, number of proposals
        by reading the h5 files
        """
        num_proposals = int(self.num_proposals[index])
        label_proposals = self.label_proposals[index]
        proposals = copy.deepcopy(label_proposals[:num_proposals, :])

        # proposal mask to filter out low-confidence proposals or backgrounds
        # mask is 1 if proposal is included
        pnt_mask = (proposals[:, 6] >= self.prop_thresh)
        if self.exclude_bgd_det:
            pnt_mask &= (proposals[:, 5] != 0)

        num_props = min(proposals.shape[0], self.max_proposals)

        padded_props = self.pad_words_with_vocab(
            proposals.tolist(), pad_len=self.max_proposals, defm=[[0]*7])
        padded_mask = self.pad_words_with_vocab(
            pnt_mask.tolist(), pad_len=self.max_proposals, defm=[0])
        return np.array(padded_props), np.array(padded_mask), num_props

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
        if self.cfg.misc.add_prop_to_region:
            region_feature = np.concatenate(
                [region_feature, props[:num_proposals, :5]],
                axis=1
            )

        # load the frame-wise segment feature
        seg_rgb_file = self.seg_feature_root / f'{vid_id_ix[2:]}_resnet.npy'
        seg_motion_file = self.seg_feature_root / f'{vid_id_ix[2:]}_bn.npy'

        assert seg_rgb_file.exists() and seg_motion_file.exists()

        seg_rgb_feature = np.load(seg_rgb_file)
        seg_motion_feature = np.load(seg_motion_file)
        seg_feature_raw = np.concatenate(
            (seg_rgb_feature, seg_motion_feature), axis=1)

        return region_feature, seg_feature_raw

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
        """
        Basically, this returns stuff for the
        vid_seg_id obtained from the idx
        """
        row = self.annots.iloc[idx]

        vid_id = row['vid_id']
        seg_id = str(row['seg_id'])
        vid_seg_id = row['id']
        ix = row['Index']

        # num_segs = self.annots[self.annots.vid_id == vid_id].seg_id.max()
        # num_segs = row['num_segs']

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

        seg_feature_for_frms, seg_feature_for_frms_glob = (
            self.get_seg_feat_for_frms(
                seg_feature_raw, timestamps, dur, idx)
        )
        # get gt annotations
        # Get the act ent annotations
        # captions = [self.anet_ent_captions[vid_id]['segments'][seg_id]]
        # ncap = len(captions)

        # gt_annot_dict = self.get_gt_annots(captions, idx)

        # # pad_gt_bboxs = gt_annot_dict['padded_gt_box']
        # # num_box = gt_annot_dict['num_box']

        # frm_mask = self.get_frm_mask(
        #     padded_props[:num_props, 4], pad_gt_bboxs[:num_box, 4])
        # pad_frm_mask = np.ones((self.max_proposals, self.max_gt_box))
        # pad_frm_mask[:num_props, :num_box] = frm_mask

        # 0 - number of captions
        # 1 - number of proposals
        # 2 - number of boxes
        # 3 - segment id
        # 4 - number of segments
        # 5 - start time stamp
        # 6 - end time stamp
        # num = torch.FloatTensor(
        #     [ncap, num_props, num_box, int(seg_id),
        #      # max(self.num_seg_per_vid[vid_id]
        #      # )+1,
        #      num_segs,
        #      timestamps[0]*1./dur,
        #      timestamps[1]*1./dur]
        # )  # 3 + 4 (seg_id, num_of_seg_in_video, seg_start_time, seg_end_time)

        pad_pnt_mask = torch.tensor(pad_pnt_mask).long()
        # pnt_mask2 = torch.cat(
        #     (pad_pnt_mask.new(1).fill_(0),
        #      pad_pnt_mask),
        #     dim=0
        # )

        pad_region_feature = np.zeros(
            (self.max_proposals, region_feature.shape[1]))
        pad_region_feature[:num_props] = region_feature[:num_props]

        out_dict = {
            'seg_feature': torch.from_numpy(seg_feature).float(),
            'seg_feature_for_frms': torch.from_numpy(
                seg_feature_for_frms).float(),
            'seg_feature_for_frms_glob': torch.from_numpy(
                seg_feature_for_frms_glob).float(),
            # 'num': num,
            # 'seq_len': torch.tensor(gt_annot_dict['seq_len']).long(),
            'num_props': torch.tensor(num_props).long(),
            # 'num_box': torch.tensor(num_box).long(),
            'pad_proposals': torch.tensor(padded_props).float(),
            # 'pad_gt_bboxs': torch.tensor(pad_gt_bboxs).float(),
            # 'pad_gt_box_mask': torch.tensor(
            # gt_annot_dict['padded_gt_box_mask']).byte(),
            'seg_id': torch.tensor(int(seg_id)).long(),
            'idx': torch.tensor(idx).long(),
            'ann_idx': torch.tensor(idx).long(),
            'pad_region_feature': torch.tensor(pad_region_feature).float(),
            # 'pad_frm_mask': torch.tensor(pad_frm_mask).byte(),
            'sample_idx': torch.tensor(sample_idx).long(),
            'pad_pnt_mask': pad_pnt_mask.byte(),
            # 'pad_pnt_mask2': pnt_mask2.byte(),
        }

        return out_dict


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/anet_srl_cfg.yml')))

    anet_ent_ds = AnetEntDataset(
        cfg=cfg, ann_file=cfg.ds.ann_file.train,
        split_type='train'
    )
