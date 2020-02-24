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
        self.trn_anet_ent_orig_file = Path(self.cfg.ds.orig_anet_ent_clss)
        assert self.trn_anet_ent_orig_file.exists()
        self.trn_anet_ent_orig_data = json.load(
            open(self.trn_anet_ent_orig_file))

        self.trn_anet_ent_preproc_file = Path(
            self.cfg.ds.preproc_anet_ent_clss)
        assert self.trn_anet_ent_preproc_file.parent.exists()

    def add_pronouns(self):
        def upd(segv):
            """
            segv: Dict.
            Keys: 'process_clss' etc
            Update the values for process_clss
            """
            proc_clss = segv['process_clss'][:]
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
            segv['process_clss'] = new_proc_clss
            return
        out_dict_vid = {}
        for vidk, vidv in tqdm(self.trn_anet_ent_orig_data['annotations'].items()):
            out_dict_seg_vid = {}
            for segk, segv in vidv['segments'].items():
                upd(segv)
                out_dict_seg_vid[segk] = segv
            out_dict_vid[vidk] = {'segments': out_dict_seg_vid}
        json.dump(out_dict_vid, open(self.trn_anet_ent_preproc_file, 'w'))


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/create_asrl_cfg.yml')))
    anet_pre = AnetEntFiles(cfg)
    anet_pre.add_pronouns()
