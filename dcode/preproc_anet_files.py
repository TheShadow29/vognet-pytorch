"""
Small preprocessing done for Anet files
In particular:
[ ] Add 'He' to 'man', 'boy', similarly for 'she' to 'woman', 'girl', 'lady'
[ ] Resize ground-truth box
"""

import json
from pathlib import Path
# import copy
from yacs.config import CfgNode as CN
import yaml
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np


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

    def run(self):
        # out_ann = self.get_vidseg_hw_map(
        # ann=self.trn_anet_ent_orig_data['annotations'])
        out_ann = self.get_vidseg_hw_map(
            ann=self.trn_anet_ent_orig_data)
        # out_ann = self.add_pronouns(out_ann)
        json.dump(out_ann, open(self.trn_anet_ent_preproc_file, 'w'))

        # self.resize_props()

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

        # with h5py.File(self.cfg.ds.proposal_h5_resized, 'w') as f:
        #     keys = [k for k in h5_proposal_file.keys()]
        #     for k in keys:
        #         if k != 'dets_labels':
        #             f.create_dataset(k, data=h5_proposal_file[k])
        #         else:
        #             f.create_dataset(k, data=label_proposals)

        return out_dict_vid

    def resize_props(self):
        vid_dict_df = pd.DataFrame(json.load(
            open(self.cfg.ds.anet_ent_split_file))['videos'])
        vid_dict_df.index.name = 'Index'
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


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/create_asrl_cfg.yml')))
    anet_pre = AnetEntFiles(cfg)
    # anet_pre.add_pronouns()
    # anet_pre.get_vidseg_hw_map()
    anet_pre.run()
    # anet_pre.resize_props()
