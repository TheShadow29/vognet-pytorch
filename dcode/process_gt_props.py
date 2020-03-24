"""
By default, we are using proposal boxes.
Instead, we only consider the gts.
"""

import numpy as np
from pathlib import Path
import h5py
import json
import pandas as pd
from tqdm import tqdm
import copy
from box_utils import box_iou
import torch
from collections import OrderedDict


class GTPropExtractor(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Assert h5 file to read from exists
        self.proposal_h5 = Path(self.cfg.ds.proposal_h5_resized)
        assert self.proposal_h5.exists()

        with h5py.File(self.proposal_h5, 'r',
                       driver='core') as h5_proposal_file:
            self.num_proposals = h5_proposal_file['dets_num'][:]
            self.label_proposals = h5_proposal_file['dets_labels'][:]

        nppf = self.cfg.ds.ngt_prop
        self.out_label_proposals = np.zeros_like(
            self.label_proposals)[:, :10*nppf, ...]
        self.out_num_proposals = np.zeros_like(self.num_proposals)

        # Assert region features exists
        self.feature_root = Path(self.cfg.ds.feature_root)
        assert self.feature_root.exists()

        # Assert act ent caption file with bbox exists
        self.anet_ent_annot_file = Path(self.cfg.ds.anet_ent_annot_file)
        assert self.anet_ent_annot_file.exists()

        if cfg.ds.ngt_prop == 5:
            self.out_dir = Path(self.cfg.ds.feature_gt5_root)
            self.out_proposal_h5 = Path(self.cfg.ds.proposal_gt5_h5)
        else:
            raise NotImplementedError

        self.out_dir.mkdir(exist_ok=True)
        # Load anet bbox
        with open(self.anet_ent_annot_file) as f:
            self.anet_ent_captions = json.load(f)

        # trn_df = pd.read_csv(self.cfg.ds.trn_ann_file)
        # val_df = pd.read_csv(self.cfg.ds.val_ann_file)

        # self.req_df = pd.concat([trn_df, val_df])

    def do_for_all_vid_seg(self, save=True):
        recall_num = 0
        recall_tot = 0
        self.cfg.no_gt_count = 0
        for row_num, vid_seg_row in tqdm(self.req_df.iterrows(),
                                         total=len(self.req_df)):
            vid_seg_id = vid_seg_row['id']
            vid_seg = vid_seg_id.split('_segment_')
            vid = vid_seg[0]
            seg = str(int(vid_seg[1]))

            annot = self.anet_ent_captions[vid]['segments'][seg]
            gt_boxs = annot['bbox']
            gt_frms = annot['frm_idx']

            prop_index = vid_seg_row['Index']
            props = copy.deepcopy(self.label_proposals[prop_index])
            num_props = int(copy.deepcopy(self.num_proposals[prop_index]))

            if num_props < 1000:
                # import pdb
                # pdb.set_trace()
                assert np.all(props[num_props:, [0, 1, 2, 3]] == 0)

            region_feature_file = self.feature_root / f'{vid_seg_id}.npy'
            # if save:
            prop_feats_load = np.load(region_feature_file)
            prop_feats = np.zeros((10, *prop_feats_load.shape[1:]))
            prop_feats[:prop_feats_load.shape[0]] = prop_feats_load
            # prop_feats = prop_feats.reshape(-1, prop_feats.shape[2]).copy()
            # prop_feats = prop_feats[:num_props, ...]
            # assert len(prop_feats) == len(props)
            # assert len(props) == num_props

            # else:
            # prop_feats = None

            out_file = self.out_dir / f'{vid_seg_id}.npy'
            # out_dict = self.do_for_one_vid_seg(
            #     props, prop_feats, gt_boxs, gt_frms, out_file,
            #     save=save
            # )
            nppf = self.cfg.ds.ngt_prop
            out_dict = self.prop10_one_vid_seg(
                props, prop_feats, gt_boxs, gt_frms, out_file,
                save=save, nppf=nppf
            )
            # out_dict = self.no_gt_prop10_one_vid_seg(
            #     props, prop_feats, gt_boxs, gt_frms, out_file,
            #     save=save
            # )

            if save:
                num_prop = out_dict['num_prop']
                self.out_label_proposals[prop_index][:num_prop] = (
                    out_dict['out_props']
                )
                self.out_num_proposals[prop_index] = num_prop

            recall_num += out_dict['recall']
            recall_tot += out_dict['num_gt']
            # if row_num > 1000:
            # break
        recall = recall_num.item() / recall_tot
        if save:
            with h5py.File(self.out_proposal_h5, 'w') as f:
                f['dets_labels'] = self.out_label_proposals
                f['dets_num'] = self.out_num_proposals
        return recall

    def prop10_one_vid_seg(self, props, prop_feats,
                           gt_boxs, gt_frms, out_file,
                           save=True, nppf=10):
        nfrms = 10
        props = torch.tensor(props).float()
        prop_feats = torch.tensor(prop_feats).float()
        # gt_frms_dict = {}
        # for gfrm, gbox in zip(gt_frms, gt_boxs):
        #     if gfrm not in gt_frms_dict:
        #         gt_frms_dict[gfrm] = []
        #     gt_frms_dict[gfrm].append(gbox)
        gt_frms_set = set(gt_frms)
        gt_boxs = torch.tensor(gt_boxs).float()
        gt_frms = torch.tensor(gt_frms).float()

        ngt = len(gt_boxs)

        nppf = nppf

        prop_frms = props[:, 4]
        frm_msk = prop_frms[:, None] == gt_frms
        if len(gt_boxs) > 0 and len(props) > 0:
            ious = box_iou(props[:, :4], gt_boxs) * frm_msk.float()
            ious_max, ious_arg_max = ious.max(dim=0)
            if len(ious_arg_max) > nppf:
                ious_arg_max = ious_arg_max[:nppf]
            out_props = props[ious_arg_max]
            out_props_inds = ious_arg_max % 100
            recall = (ious_max > 0.5).sum()
        else:
            self.cfg.no_gt_count += 1
            ngt = 1
            recall = 0
            ious = torch.zeros(props.size(0), 1)
            out_props = props[0]
            out_props_inds = torch.tensor(0)

        fin_out_props = {}
        props1 = props.view(10, 100, 7)
        prop_dim = prop_feats.size(-1)
        prop_feats1 = prop_feats.view(10, 100, prop_dim)

        for frm in range(nfrms):
            if frm not in fin_out_props:
                fin_out_props[frm] = []

            if frm in gt_frms_set:
                props_inds_gt_in_frm = out_props_inds[out_props[..., 4] == frm]
                fin_out_props[frm] += props_inds_gt_in_frm.tolist()

            props_to_use_inds = props1[frm, ..., 6].argsort(descending=True)[
                :nppf]
            # props_to_use_inds = np.random.choice(
            #     np.arange(100), size=10, replace=False
            # )
            fin_out_props[frm] += props_to_use_inds.tolist()

            fin_out_props[frm] = list(
                OrderedDict.fromkeys(fin_out_props[frm]))[:nppf]

        props_output = torch.zeros(10, nppf, 7)
        prop_feats_output = torch.zeros(10, nppf, prop_dim)

        for frm in fin_out_props:
            inds = fin_out_props[frm]
            props_output[frm] = props1[frm][inds]
            prop_feats_output[frm] = prop_feats1[frm][inds]

        props_output = props_output.view(10*nppf, 7).detach().cpu().numpy()
        prop_feats_output = prop_feats_output.view(
            10, nppf, prop_dim).detach().cpu().numpy()

        if save:
            np.save(out_file, prop_feats_output)

        return {
            'out_props': props_output,
            'recall': recall,
            'num_prop': 100,
            'num_gt': ngt
        }

    def no_gt_prop10_one_vid_seg(self, props, prop_feats,
                                 gt_boxs, gt_frms, out_file,
                                 save=True):
        nfrms = 10
        props = torch.tensor(props).float()
        prop_feats = torch.tensor(prop_feats).float()
        # gt_frms_dict = {}
        # for gfrm, gbox in zip(gt_frms, gt_boxs):
        #     if gfrm not in gt_frms_dict:
        #         gt_frms_dict[gfrm] = []
        #     gt_frms_dict[gfrm].append(gbox)
        gt_frms_set = set(gt_frms)
        gt_boxs = torch.tensor(gt_boxs).float()
        gt_frms = torch.tensor(gt_frms).float()

        ngt = len(gt_boxs)

        nppf = 100

        fin_out_props = {}
        props1 = props.view(10, 100, 7)
        prop_dim = prop_feats.size(-1)
        prop_feats1 = prop_feats.view(10, 100, prop_dim)

        for frm in range(nfrms):
            if frm not in fin_out_props:
                fin_out_props[frm] = []

            # if frm in gt_frms_set:
            #     props_inds_gt_in_frm = out_props_inds[out_props[..., 4] == frm]
            #     fin_out_props[frm] += props_inds_gt_in_frm.tolist()
            props_to_use_inds = props1[frm, ..., 6].argsort(descending=True)[
                :nppf]
            fin_out_props[frm] += props_to_use_inds.tolist()

            fin_out_props[frm] = list(
                OrderedDict.fromkeys(fin_out_props[frm]))[:nppf]

        props_output = torch.zeros(10, nppf, 7)
        prop_feats_output = torch.zeros(10, nppf, prop_dim)

        for frm in fin_out_props:
            inds = fin_out_props[frm]
            props_output[frm] = props1[frm][inds]
            prop_feats_output[frm] = prop_feats1[frm][inds]

        props_output = props_output.view(nfrms * nppf, 7)
        prop_feats_output = prop_feats_output.view(
            nfrms, nppf, prop_dim).detach().cpu().numpy()

        if len(gt_boxs) > 0 and len(props_output) > 0:
            prop_frms = props_output[:, 4]
            frm_msk = prop_frms[:, None] == gt_frms
            ious = box_iou(props_output[:, :4], gt_boxs) * frm_msk.float()
            ious_max, ious_arg_max = ious.max(dim=0)
            recall = (ious_max > 0.5).sum()
        else:
            self.cfg.no_gt_count += 1
            ngt = 1
            recall = 0
            ious = torch.zeros(props.size(0), 1)

        props_output = props_output.detach().cpu().numpy()

        if save:
            np.save(out_file, prop_feats_output)

        return {
            'out_props': props_output,
            'recall': recall,
            'num_prop': 100,
            'num_gt': ngt
        }

    def do_for_one_vid_seg(self, props, prop_feats,
                           gt_boxs, gt_frms, out_file,
                           save=True):
        """
        props: all the proposal boxes
        gt_boxs: all the groundtruth_boxes
        out_props: props with highest IoU with gt_box
        # nframes x 1,
        one-to-one correspondence
        Also, used to calculate recall.
        """
        props = torch.tensor(props).float()
        gt_boxs = torch.tensor(gt_boxs).float()
        gt_frms = torch.tensor(gt_frms).float()

        ngt = len(gt_boxs)

        prop_frms = props[:, 4]
        frm_msk = prop_frms[:, None] == gt_frms

        if len(gt_boxs) > 0 and len(props) > 0:
            ious = box_iou(props[:, :4], gt_boxs) * frm_msk.float()
            ious_max, ious_arg_max = ious.max(dim=0)
            recall = (ious_max > 0.5).sum().float()
            out_props = props[ious_arg_max]
        else:
            self.cfg.no_gt_count += 1
            ngt = 1
            recall = 0
            ious = torch.zeros(props.size(0), 1)
            out_props = props[0]

        nprop = ngt
        if save:
            prop_dim = prop_feats.size(-1)
            prop_feats = torch.tensor(prop_feats).float()
            out_prop_feats = prop_feats[ious_arg_max].view(
                1, ngt, prop_dim).detach().cpu().numpy()
            assert list(out_prop_feats.shape[:2]) == [1, ngt]
            np.save(out_file, out_prop_feats)

        return {
            'out_props': out_props,
            'recall': recall,
            'num_prop': nprop,
            'num_gt': ngt
        }


if __name__ == '__main__':
    from extended_config import cfg as conf
    cfg = conf
    gtp = GTPropExtractor(cfg)
    recall = gtp.do_for_all_vid_seg(save=True)
    print(recall)
