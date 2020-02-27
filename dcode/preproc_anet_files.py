"""
Small preprocessing done for Anet files
In particular:
[ ] Add 'He' to 'man', 'boy', similarly for 'she' to 'woman', 'girl', 'lady'
[ ] Resize ground-truth box
"""

import json
from pathlib import Path
from yacs.config import CfgNode as CN
import yaml
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
from utils.box_utils import box_iou
import copy
import torch
from collections import OrderedDict


class AnetEntFiles:
    def __init__(self, cfg):
        self.cfg = cfg
        self.conv_dict = {
            'man': 'he',
            'boy': 'he',
            'woman': 'she',
            'girl': 'she',
            'lady': 'she'
        }
        self.open_req_files()

    def open_req_files(self):
        # self.trn_anet_ent_orig_file = Path(self.cfg.ds.orig_anet_ent_clss)
        self.trn_anet_ent_orig_file = Path(self.cfg.ds.anet_ent_annot_file)
        assert self.trn_anet_ent_orig_file.exists()
        self.trn_anet_ent_orig_data = json.load(
            open(self.trn_anet_ent_orig_file))

        self.trn_anet_ent_preproc_file = Path(
            self.cfg.ds.preproc_anet_ent_clss)
        assert self.trn_anet_ent_preproc_file.parent.exists()

        self.vid_dict_df = pd.DataFrame(json.load(
            open(self.cfg.ds.anet_ent_split_file))['videos'])
        self.vid_dict_df.index.name = 'Index'

        # Assert region features exists
        self.feature_root = Path(self.cfg.ds.feature_root)
        assert self.feature_root.exists()

        self.feature_root_gt5 = Path(self.cfg.ds.feature_gt5_root)
        self.feature_root_gt5.mkdir(exist_ok=True)
        assert self.feature_root_gt5.exists()

    def run(self):
        # out_ann = self.get_vidseg_hw_map(
        # ann=self.trn_anet_ent_orig_data['annotations'])
        out_ann = self.get_vidseg_hw_map(
            ann=self.trn_anet_ent_orig_data)
        # out_ann = self.add_pronouns(out_ann)
        json.dump(out_ann, open(self.trn_anet_ent_preproc_file, 'w'))

    def add_pronouns(self, ann):
        def upd(segv):
            """
            segv: Dict.
            Keys: 'process_clss' etc
            Update the values for process_clss
            """
            pck = 'process_clss'
            if pck not in segv:
                pck = 'clss'
                assert pck in segv
            proc_clss = segv[pck][:]
            assert isinstance(proc_clss, list)
            if len(proc_clss) == 0:
                return
            assert isinstance(proc_clss[0], list)
            new_proc_clss = []
            for pc in proc_clss:
                new_pc = []
                for p in pc:
                    if p in self.conv_dict:
                        new_pc.append(p)
                        new_pc.append(self.conv_dict[p])
                    else:
                        new_pc.append(p)
                new_proc_clss.append(new_pc)
            segv[pck] = new_proc_clss
            return
        out_dict_vid = {}
        for vidk, vidv in tqdm(ann.items()):
            out_dict_seg_vid = {}
            for segk, segv in vidv['segments'].items():
                upd(segv)
                out_dict_seg_vid[segk] = segv
            out_dict_vid[vidk] = {'segments': out_dict_seg_vid}

        return out_dict_vid

    def get_vidseg_hw_map(self, ann=None):
        def upd(segv, sw, sh):
            """
            segv: Dict
            Change process_bnd_box wrt hw
            """
            pbk = 'process_bnd_box'
            if pbk not in segv:
                pbk = 'bbox'
                assert pbk in segv
            if len(segv[pbk]) == 0:
                return
            process_bnd_box = np.array(
                segv[pbk][:]).astype(float)
            process_bnd_box[:, [0, 2]] *= sw
            process_bnd_box[:, [1, 3]] *= sh
            process_bnd_box = process_bnd_box.astype(int)
            segv[pbk] = process_bnd_box.tolist()
            return

        vid_dict_df = self.vid_dict_df

        h5_proposal_file = h5py.File(
            self.cfg.ds.proposal_h5, 'r', driver='core')

        # num_proposals = h5_proposal_file['dets_num'][:]
        # label_proposals = h5_proposal_file['dets_labels'][:]

        hw_vids = h5_proposal_file['hw'][:].astype(float).tolist()
        out_dict = {}
        for row_ind, row in tqdm(vid_dict_df.iterrows()):
            vid_id = row['vid_id']
            if vid_id not in out_dict:
                out_dict[vid_id] = hw_vids[row_ind]
            else:
                hw = hw_vids[row_ind]
                if not hw == [0., 0.]:
                    assert hw == out_dict[vid_id]
        json.dump(out_dict, open(self.cfg.ds.vid_hw_map, 'w'))

        nw = self.cfg.ds.resized_width
        nh = self.cfg.ds.resized_height
        out_dict_vid = {}
        for vidk, vidv in tqdm(ann.items()):
            out_dict_seg_vid = {}
            oh, ow = out_dict[vidk]
            if ow != 0. or oh != 0.:
                sw = nw / ow
                sh = nh / oh
            else:
                sw, sh = 1., 1.
            for segk, segv in vidv['segments'].items():
                upd(segv, sw*1., sh*1.)
                out_dict_seg_vid[segk] = segv
            out_dict_vid[vidk] = {'segments': out_dict_seg_vid}

        return out_dict_vid

    def resize_props(self):
        h5_proposal_file = h5py.File(
            self.cfg.ds.proposal_h5, 'r', driver='core')

        hw_vids = h5_proposal_file['hw'][:].astype(float).tolist()
        label_proposals = h5_proposal_file['dets_labels'][:]

        nw = self.cfg.ds.resized_width
        nh = self.cfg.ds.resized_height

        for row_ind in tqdm(range(len(label_proposals))):
            oh, ow = hw_vids[row_ind]
            if ow != 0. or oh != 0.:
                sw = nw / ow
                sh = nh / oh
            else:
                sw, sh = 1., 1.

            label_proposals[row_ind, :, [0, 2]] *= sw
            label_proposals[row_ind, :, [1, 3]] *= sh
        with h5py.File(self.cfg.ds.proposal_h5_resized, 'w') as f:
            keys = [k for k in h5_proposal_file.keys()]
            for k in keys:
                if k != 'dets_labels':
                    f.create_dataset(k, data=h5_proposal_file[k])
                else:
                    f.create_dataset(k, data=label_proposals)

        return

    def choose_gt5(self, save=True):
        """
        Choose 5 proposals for each frame
        """
        h5_proposal_file = h5py.File(
            self.cfg.ds.proposal_h5_resized, 'r', driver='core')
        # h5_proposal_file = h5py.File(
        # self.cfg.ds.proposal_h5, 'r', driver='core')

        nppf_orig = 100
        nppf = self.cfg.ds.ngt_prop
        nfrms = self.cfg.ds.num_frms
        # Note these are resized labels
        label_proposals = h5_proposal_file['dets_labels'][:]
        num_proposals = h5_proposal_file['dets_num'][:]
        out_label_proposals = np.zeros_like(
            label_proposals)[:, :nfrms*nppf, ...]
        out_num_proposals = np.zeros_like(num_proposals)
        vid_dict_df = self.vid_dict_df

        anet_ent_preproc_data = json.load(open(self.trn_anet_ent_preproc_file))
        # anet_ent_preproc_data = json.load(
        # open(self.cfg.ds.anet_ent_annot_file))

        recall_num = 0
        recall_tot = 0

        for row_ind, row in tqdm(vid_dict_df.iterrows(),
                                 total=len(vid_dict_df)):
            # if row_ind > 1000:
            # break
            vid = row['vid_id']
            seg = row['seg_id']
            vid_seg_id = row['id']

            annot = anet_ent_preproc_data[vid]['segments'][seg]
            gt_boxs = annot['bbox']
            gt_frms = annot['frm_idx']

            prop_index = row_ind

            props = copy.deepcopy(label_proposals[prop_index])
            num_props = int(copy.deepcopy(num_proposals[prop_index]))

            if num_props < nfrms * nppf_orig:
                # import pdb
                # pdb.set_trace()
                assert np.all(props[num_props:, [0, 1, 2, 3]] == 0)

            region_feature_file = self.feature_root / f'{vid_seg_id}.npy'
            if not region_feature_file.exists():
                continue
            prop_feats_load = np.load(region_feature_file)
            prop_feats = np.zeros((nfrms, *prop_feats_load.shape[1:]))
            prop_feats[:prop_feats_load.shape[0]] = prop_feats_load

            out_file = self.feature_root_gt5 / f'{vid_seg_id}.npy'
            out_dict = self.choose_gt5_for_one_vid_seg(
                props, prop_feats, gt_boxs, gt_frms, out_file,
                save=save, nppf=nppf, nppf_orig=nppf_orig, nfrms=nfrms
            )

            if save:
                num_prop = out_dict['num_prop']
                out_label_proposals[prop_index][:num_prop] = (
                    out_dict['out_props']
                )
                out_num_proposals[prop_index] = num_prop

            recall_num += out_dict['recall']
            recall_tot += out_dict['num_gt']

        recall = recall_num.item() / recall_tot
        print(f'Recall is {recall}')
        if save:
            with h5py.File(self.cfg.ds.proposal_gt5_h5_resized, 'w') as f:
                keys = [k for k in h5_proposal_file.keys()]
                keys.remove('dets_labels')
                keys.remove('dets_num')
                for k in keys:
                    f.create_dataset(k, data=h5_proposal_file[k])

                f.create_dataset('dets_labels', data=out_label_proposals)
                f.create_dataset('dets_num', data=out_num_proposals)

        return recall

    def choose_gt5_for_one_vid_seg(
            self, props, prop_feats,
            gt_boxs, gt_frms, out_file,
            save=True, nppf=5, nppf_orig=100, nfrms=10):
        """
        Choose for 5 props per frame
        """
        # Convert to torch tensors for box_iou computations
        # props: 10*100 x 7
        props = torch.tensor(props).float()
        prop_feats = torch.tensor(prop_feats).float()
        # set for comparing
        gt_frms_set = set(gt_frms)
        gt_boxs = torch.tensor(gt_boxs).float()
        gt_frms = torch.tensor(gt_frms).float()

        # Get the frames for the proposal boxes are
        prop_frms = props[:, 4]
        # Create a frame mask.
        # Basically, if the iou = 0 if the proposal and
        # the ground truth box lie in different frames
        frm_msk = prop_frms[:, None] == gt_frms
        if len(gt_boxs) > 0 and len(props) > 0:
            ious = box_iou(props[:, :4], gt_boxs) * frm_msk.float()
            # get the max iou proposal for each bounding box
            ious_max, ious_arg_max = ious.max(dim=0)
            # if len(ious_arg_max) > nppf:
            # ious_arg_max = ious_arg_max[:nppf]
            out_props = props[ious_arg_max]
            out_props_inds = ious_arg_max % 100
            recall = (ious_max > 0.5).sum()
            ngt = len(gt_boxs)
        else:
            ngt = 1
            recall = 0
            ious = torch.zeros(props.size(0), 1)
            out_props = props[0]
            out_props_inds = torch.tensor(0)

        # Dictionary to store final proposals to use
        fin_out_props = {}
        # Reshape proposals and proposal features to
        # nfrms x nppf x ndim
        props1 = props.view(nfrms, nppf_orig, 7)
        prop_dim = prop_feats.size(-1)
        prop_feats1 = prop_feats.view(nfrms, nppf_orig, prop_dim)

        # iterate over each frame
        for frm in range(nfrms):
            if frm not in fin_out_props:
                fin_out_props[frm] = []

            # if there are gt boxes in the frame
            # consider the proposals which have highest iou
            # in the frame
            if frm in gt_frms_set:
                props_inds_gt_in_frm = out_props_inds[out_props[..., 4] == frm]
                # add highest iou props to the dict key
                fin_out_props[frm] += props_inds_gt_in_frm.tolist()

            # sort by their scores, and choose nppf=5 such props
            props_to_use_inds = props1[frm, ..., 6].argsort(descending=True)[
                :nppf]
            # add 5 such props to the list
            fin_out_props[frm] += props_to_use_inds.tolist()

            # Restrict the total to 5
            fin_out_props[frm] = list(
                OrderedDict.fromkeys(fin_out_props[frm]))[:nppf]

        # Saving them, init with zeros
        props_output = torch.zeros(nfrms, nppf, 7)
        prop_feats_output = torch.zeros(nfrms, nppf, prop_dim)

        # set for each frame
        for frm in fin_out_props:
            inds = fin_out_props[frm]
            props_output[frm] = props1[frm][inds]
            prop_feats_output[frm] = prop_feats1[frm][inds]

        # Reshape nfrm x nppf x ndim -> nfrm*nppf x ndim
        props_output = props_output.view(nfrms*nppf, 7).detach().cpu().numpy()
        prop_feats_output = prop_feats_output.view(
            nfrms, nppf, prop_dim).detach().cpu().numpy()

        if save:
            np.save(out_file, prop_feats_output)

        return {
            'out_props': props_output,
            'recall': recall,
            'num_prop': nppf*nfrms,
            'num_gt': ngt
        }

    def compute_recall(self):
        """
        Compute recall for the created h5 file
        """
        with h5py.File(self.cfg.ds.proposal_gt5_h5_resized, 'r') as f:
            label_proposals = f['dets_labels'][:]

        vid_dict_df = self.vid_dict_df

        anet_ent_preproc_data = json.load(open(self.trn_anet_ent_preproc_file))

        recall_num = 0
        recall_tot = 0

        for row_ind, row in tqdm(vid_dict_df.iterrows(),
                                 total=len(vid_dict_df)):

            vid = row['vid_id']
            seg = row['seg_id']
            vid_seg_id = row['id']

            annot = anet_ent_preproc_data[vid]['segments'][seg]
            gt_boxs = torch.tensor(annot['bbox']).float()
            gt_frms = annot['frm_idx']

            prop_index = row_ind

            region_feature_file = self.feature_root / f'{vid_seg_id}.npy'
            if not region_feature_file.exists():
                continue

            props = copy.deepcopy(label_proposals[prop_index])
            props = torch.tensor(props).float()
            # props = props.view(10, -1, 7)

            for fidx, frm in enumerate(gt_frms):
                prop_frms = props[props[..., 4] == frm]
                gt_box_in_frm = gt_boxs[fidx]

                ious = box_iou(prop_frms[:, :4], gt_box_in_frm)

                ious_max, ious_arg_max = ious.max(dim=0)
                # conversion to long is important, otherwise
                # after 256 becomes 0
                recall_num += (ious_max > 0.5).any().long()

            recall_tot += len(gt_boxs)

        recall = recall_num.item() / recall_tot
        print(f'Recall is {recall}')
        return


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/create_asrl_cfg.yml')))
    anet_pre = AnetEntFiles(cfg)
    # anet_pre.compute_recall()
    # anet_pre.choose_gt5(save=True)
    # anet_pre.add_pronouns()
    # anet_pre.get_vidseg_hw_map()
    # anet_pre.run()
    # anet_pre.resize_props()
