"""
Visualize Predictions
"""

import pandas as pd
import pickle
from PIL import Image
from pathlib import Path
from eval_fn_corr import (
    GroundEval_SEP,
    GroundEval_TEMP,
    GroundEval_SPAT
)
import fire
from munch import Munch
from typing import List


class ASRL_Vis:
    def open_required_files(self, ann_file):
        self.annots = pd.read_csv(ann_file)

    def draw_boxes_all_indices(self, preds):
        # self.preds
        pass

    def prepare_img(self, img_list: List):
        """
        How to concate the image from an image list
        """
        raise NotImplementedError

    def extract_bbox_per_frame(self, preds):
        """
        Obtain the bounding boxes for each frame
        """
        raise NotImplementedError

    def all_inds(self, pred_file, split_type):
        self.prepare_gt(split_type)

    def draw_boxes_one_index(
            self, pred, gt_row, conc_type
    ):
        frm_tdir = Path('/home/Datasets/ActNetEnt/frames_10frm/')
        vid_file_id_list = pred['idx_vid']

        rows = self.annots.iloc[vid_file_id_list]
        vid_seg_id_list = rows['id']

        img_file_dict = {
            k: sorted(
                [x for x in (frm_tdir/k).iterdir()],
                key=lambda x: int(x.stem)
            )
            for k in vid_seg_id_list
        }
        img_list_dict = {
            k: [Image.open(img_file) for img_file in img_file_list]
            for k, img_file_list in img_file_dict.items()
        }

        img = self.prepare_img(img_list_dict)
        pass


class ASRL_Vis_SEP(GroundEval_SEP, ASRL_Vis):
    pass


class ASRL_Vis_TEMP(GroundEval_TEMP, ASRL_Vis):
    pass


class ASRL_Vis_SPAT(GroundEval_SPAT, ASRL_Vis):
    pass


def main(pred_file, split_type='valid', **kwargs):
    if 'cfg' not in kwargs:
        from extended_config import (
            cfg as conf,
            key_maps,
            # CN,
            update_from_dict,
            # post_proc_config
        )
        cfg = conf
        cfg = update_from_dict(cfg, kwargs, key_maps)
    else:
        cfg = kwargs['cfg']
        cfg.freeze()
    # grnd_eval = GroundEval_Corr(cfg)
    # grnd_eval = GroundEvalDS4(cfg)
    comm = Munch()
    exp = cfg.ds.exp_setting
    if exp == 'gt5':
        comm.num_prop_per_frm = 5
    elif exp == 'p100':
        comm.num_prop_per_frm = 100
    else:
        raise NotImplementedError

    conc_type = cfg.ds.conc_type
    if conc_type == 'sep' or conc_type == 'svsq':
        avis = ASRL_Vis_SEP(cfg, comm)
    elif conc_type == 'temp':
        avis = ASRL_Vis_TEMP(cfg, comm)
    elif conc_type == 'spat':
        avis = ASRL_Vis_SPAT(cfg, comm)
    else:
        raise NotImplementedError

    # avis.draw_boxes_all_indices(
    #     pred_file, split_type=split_type
    # )

    return avis


if __name__ == '__main__':
    fire.Fire(main)
