"""
Corrected version of eval_fn
Some differences in computing the correctness

In eval_fn we only use annotated frames
and require that box is correct only in
those frames.

The above is fine if only one video
However, in multi-video setting, it is not correct

Instead we do the following:
- Sep: Require a score for the correlation with
the whole video.
- Temporal: For frames in the correct video, only consider
the annotated once. For incorrect video, none should
be considered as correct.
- Spatial: For annotated frames, require correct output.
For other frames, require answer lies in the correct
video.
"""

import _init_stuff
import pandas as pd
import torch
import numpy as np
import pickle
import json
from tqdm import tqdm
from munch import Munch
import ast
from box_utils import box_iou
import numpy as np
import fire
from collections import Counter


def list_of_dicts_avg(lst_dict):
    return pd.DataFrame(lst_dict).mean().to_dict()


class GroundEval_Corr:
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        self.res_dicts = ['res_dict']
        self.prob_thresh = self.cfg.train.prob_thresh
        self.prepare_gt(split_type='valid')
        self.after_init()

    def after_init(self):
        return

    def prepare_gt(self, split_type='valid'):
        # self.srl_annots1 = pd.read_csv(self.cfg.ds.val_verb_ent_file)
        self.srl_annots1 = pd.read_csv(self.cfg.ds.val_ds4_inds)
        assert hasattr(self, 'srl_annots1')
        for k in self.srl_annots1.columns:
            first_word = self.srl_annots1.iloc[0][k]
            if isinstance(first_word, str) and first_word[0] == '[':
                self.srl_annots1[k] = self.srl_annots1[k].apply(
                    lambda x: ast.literal_eval(x))

        if split_type == 'valid' or split_type == 'test':
            self.annots = pd.read_csv(self.cfg.ds.val_ann_file)
            with open(self.cfg.ds.anet_ent_annot_file) as f:
                self.anet_annots = json.load(f)

            vt_split = 'val' if split_type == 'valid' else 'test'
            self.srl_annots = self.srl_annots1[
                self.srl_annots1.vt_split == vt_split]
        else:
            raise NotImplementedError

    def prepare_preds(self, predict_file):
        with open(predict_file, 'rb') as f:
            out_df = pd.DataFrame(pickle.load(f))
        return out_df.drop_duplicates(subset='idx_sent')

    def get_req_pred_from_row(self, pred_row, gt_row,
                              gt_row_ind, key='pred_boxes'):
        if not self.cfg.ds.do_ds4:
            verb_ind = [x_ind for x_ind, x in enumerate(
                pred_row.idx_verb[:pred_row.num_verbs.item()])
                if x == gt_row_ind]

            # verb was never passed
            if len(verb_ind) == 0:
                return -1

            assert len(verb_ind) == 1
            verb_ind = verb_ind[0]
            num_srl_args = len(gt_row.req_args)
            pred_boxes_for_verb = pred_row.pred_boxes[verb_ind][:num_srl_args]
            return pred_boxes_for_verb
        else:
            assert self.cfg.ds.do_ds4
            num_srl_args = len(gt_row.req_args)
            return pred_row[key][:num_srl_args]

    def eval_one_sent_idx(self, pred_row, gt_rows):
        assert len(gt_rows) == 1
        # if len(gt_rows) == 1:
        gt_row = gt_rows.iloc[0]
        gt_row_ind = gt_row.name

        results_dict = {}
        tot_dict = {}
        considered_boxes = []
        vid_seg = gt_row.vid_seg
        vid, seg = vid_seg.split('_segment_')
        seg = str(int(seg))

        anet_ann_row = self.anet_annots[vid]['segments'][seg]
        all_gt_boxes = torch.tensor(anet_ann_row['bbox'])
        all_gt_frames = torch.tensor(anet_ann_row['frm_idx'])
        assert len(all_gt_boxes) == len(all_gt_frames)

        pred_boxes_for_verb = self.get_req_pred_from_row(
            pred_row, gt_row, gt_row_ind
        )
        if pred_boxes_for_verb == -1:
            return -1

        for srl_ind, (
                srl_arg,
                srl_arg_box_indicator,
                srl_arg_box_ind
        ) in enumerate(gt_row.req_cls_pats_mask):

            if srl_arg_box_indicator == 1:
                if gt_row_ind not in results_dict:
                    results_dict[gt_row_ind] = 0
                if gt_row_ind not in tot_dict:
                    tot_dict[gt_row_ind] = 0

                tot_dict[gt_row_ind] += 1
                if srl_ind >= len(pred_boxes_for_verb):
                    continue

                box_inds = torch.tensor(srl_arg_box_ind)
                gt_boxes = torch.index_select(all_gt_boxes, 0, box_inds)
                frm_idxs = torch.index_select(all_gt_frames, 0, box_inds)
                pred_boxes = pred_boxes_for_verb[srl_ind]
                for frm_idx_ind, frm_idx in enumerate(frm_idxs):
                    predicted_box = torch.tensor(pred_boxes[frm_idx][:4])
                    groundtruth_box = gt_boxes[frm_idx_ind]
                    iou = box_iou(predicted_box.float(),
                                  groundtruth_box.float())
                    considered_boxes.append({
                        'predicted_box': predicted_box,
                        'gt_box': groundtruth_box,
                        'frm_idx': frm_idx,
                        'srl_ind': srl_ind,
                        'iou': iou
                    })
                    if iou > 0.5:
                        results_dict[gt_row_ind] += 1

        return {
            'res_dict': results_dict,
            'tot_dict': tot_dict,
            'considered_boxes': considered_boxes
        }

    def init_res_dicts(self):
        res_dicts = {k: {} for k in self.res_dicts}
        tot_dict = {}
        return res_dicts, tot_dict

    def update_res_dicts(self, res_dicts, tot_dict,
                         out, gt_row_ind):
        for res_dc_name in self.res_dicts:
            res_dicts[res_dc_name][gt_row_ind] = (
                out[res_dc_name][gt_row_ind])
            tot_dict[gt_row_ind] = out['tot_dict'][gt_row_ind]

        return

    def compute_avgs_using_res(self, res_dicts, tot_dict):
        tot_keys = sorted(list(tot_dict.keys()))
        res_dicts_np = {res_dc_name: np.array(
            [res_dicts[res_dc_name][k] for k in tot_keys])
            for res_dc_name in self.res_dicts}
        tot_np = np.array([tot_dict[k] for k in tot_keys])

        # results_np = res_dicts_np[self.res_dicts[0]]
        avg1 = {k: res_dicts_np[k].sum() / tot_np.sum()
                for k in self.res_dicts}
        avg2 = {k: np.divide(res_dicts_np[k], tot_np).mean()
                for k in self.res_dicts}

        return avg1, avg2

    def post_proc_final(self, out_dict):
        return out_dict

    def eval_ground_acc(self, predict_file, split_type='valid'):
        """
        predictions: List[Dict] / DataFrame
        groundtruths: List[Dict] / DataFrame
        """
        self.prepare_gt(split_type)
        pred_df = self.prepare_preds(predict_file)
        gt_df = self.srl_annots
        res_dicts = {k: {} for k in self.res_dicts}

        tot_dict = {}
        tot = 0

        classwise_dict = {}

        pred_df1 = pred_df.set_index('idx_sent')

        for gt_row_ind, gt_row in tqdm(gt_df.iterrows(), total=len(gt_df)):
            tot += 1
            ann_ind = gt_row.ann_ind
            pred_row = pred_df1.loc[gt_row_ind]

            targ_cmp = pred_row.targ_cmp
            verb_ids = pred_row.idx_verbs
            assert gt_row_ind == verb_ids[targ_cmp]
            gt_rows = self.srl_annots1.loc[verb_ids]
            out = self.eval_one_sent_idx(pred_row, gt_rows)
            if out != -1:
                if gt_row_ind in out['tot_dict']:
                    # Update the default res_dict
                    self.update_res_dicts(res_dicts, tot_dict, out, gt_row_ind)
                    lemma_verb = gt_row.lemma_verb
                    if lemma_verb not in classwise_dict:
                        classwise_dict[lemma_verb] = self.init_res_dicts()
                    # Update for each class
                    self.update_res_dicts(
                        classwise_dict[lemma_verb][0],
                        classwise_dict[lemma_verb][1],
                        out, gt_row_ind
                    )

        res_dicts_avg1, res_dicts_avg2 = self.compute_avgs_using_res(
            res_dicts, tot_dict
        )
        res_key_to_use = self.res_dicts[0]

        cls_avg = {k: self.compute_avgs_using_res(
            v[0], v[1]) for k, v in classwise_dict.items()}

        macro_res1, macro_res2 = zip(*[v for k, v in cls_avg.items()])

        macro_avg1 = list_of_dicts_avg(macro_res1)
        macro_avg2 = list_of_dicts_avg(macro_res2)

        out_dict = {
            'avg1': res_dicts_avg1[res_key_to_use],
            'avg2': res_dicts_avg2[res_key_to_use],
            # 'cls_avg': cls_avg,
            'macro_avg1': macro_avg1[res_key_to_use],
            'macro_avg2': macro_avg2[res_key_to_use],
            'res_dicts_avg1': res_dicts_avg1,
            'res_dicts_macro_avg1': macro_avg1,
            # 'res_dict': res_dicts,
            # 'tot_dict': tot_dict,
            # 'class_avg1':
            # 'results_np': results_np,
            # 'tot_np': tot_np,
            # 'all_res_dict_np': res_dicts_np,
            'classwise_dict': classwise_dict,
            'gt_df': gt_df
        }
        return self.post_proc_final(out_dict)


class GroundEval_SEP(GroundEval_Corr):
    def after_init(self):
        self.res_dicts = ['res_dict', 'cons_dict',
                          'vidf_dict', 'strict_res_dict']
        self.num_sampled_frm = self.cfg.ds.num_sampled_frm
        self.num_prop_per_frm = self.comm.num_prop_per_frm

    def post_proc(self, out_dict):
        key_list = list(out_dict['res_dict'].keys())
        out_dict['strict_res_dict'] = {
            k: (out_dict['res_dict'][k] == out_dict['tot_dict']
                [k]) * out_dict['tot_dict'][k]
            for k in key_list
        }
        return out_dict

    def post_proc_final(self, out_dict):
        res_dicts_avg1 = out_dict['res_dicts_avg1']
        res_dicts_macro_avg1 = out_dict['res_dicts_macro_avg1']

        out_dict['avg1_cons'] = res_dicts_avg1['cons_dict']
        out_dict['macro_avg1_cons'] = res_dicts_macro_avg1['cons_dict']

        out_dict['avg1_strict'] = res_dicts_avg1['strict_res_dict']
        out_dict['macro_avg1_strict'] = res_dicts_macro_avg1['strict_res_dict']

        out_dict['avg1_vidf'] = res_dicts_avg1['vidf_dict']
        out_dict['macro_avg1_vidf'] = res_dicts_macro_avg1['vidf_dict']
        return out_dict

    def compute_one_srl(self, pred_cmp, pred_boxes_for_srl,
                        pred_scores_for_srl,
                        targ_cmp, gt_boxes_with_frames,
                        gt_frames_all, cmp_msk):
        """
        For sep
        pred_cmp: is the chosen video (1)
        targ_cmp: is the target video (1)
        pred_boxes_for_srl: predicted boxes for
        given srl (#nvids x #nframes x #1-prop(4))
        """
        # nvids = len(pred_boxes_for_srl)
        # nfrms = len(pred_boxes_for_srl[0])
        if pred_cmp == targ_cmp:
            gt_frms = gt_boxes_with_frames[:, -1].long().tolist()
            pred_boxes = pred_boxes_for_srl[pred_cmp]
            pred_scores = pred_scores_for_srl[pred_cmp]
            for frm_idx_ind, frm_idx in enumerate(gt_frms):
                predicted_box = torch.tensor(pred_boxes[frm_idx][:4])
                pbox = torch.tensor(pred_boxes[frm_idx])
                groundtruth_box = gt_boxes_with_frames[frm_idx_ind][:4]
                prediction_score = pred_scores[frm_idx]
                assert gt_boxes_with_frames[frm_idx_ind][4] == frm_idx
                iou = box_iou(
                    predicted_box.float(),
                    groundtruth_box.float()
                )
                # TODO: Check why prediction scores
                # are ridiculously low!!
                if iou > 0.5 and prediction_score > self.prob_thresh:
                    return {
                        'targ_cmp': targ_cmp,
                        'pred_cmp': pred_cmp,
                        'predicted_box': predicted_box,
                        'pbox': pbox,
                        'gt_box': groundtruth_box,
                        'frm_idx': frm_idx,
                        'iou': iou
                    }

        return {
            'targ_cmp': targ_cmp,
            'pred_cmp': pred_cmp,
            'iou': torch.tensor(0)
        }

    def collect_box_frames_from_gt_row(self, gt_row):
        vid_seg = gt_row.vid_seg
        vid, seg = vid_seg.split('_segment_')
        seg = str(int(seg))

        anet_ann_row = self.anet_annots[vid]['segments'][seg]
        all_gt_boxes = torch.tensor(anet_ann_row['bbox'])
        all_gt_frames = torch.tensor(anet_ann_row['frm_idx'])
        assert len(all_gt_boxes) == len(all_gt_frames)
        return all_gt_boxes, all_gt_frames

    def compute_cons_vidf(self, considered_list):
        """
        considered list: List[Dict] of required stuff
        """
        cons = 1
        if len(considered_list) > 0:
            c0 = considered_list[0]
            vid_cor = c0['pred_cmp'] == c0['targ_cmp']
            return cons, vid_cor
        else:
            return 0, 0

    def pred_cmp_to_pass(self, p0_cons, p1):
        return p0_cons

    def eval_one_sent_idx(self, pred_row, gt_rows):
        """
        pred_row: boxes, scores must be of the form
        # Nvids x #Nframes x #Nprops
        """
        targ_cmp = pred_row.targ_cmp
        gt_row = gt_rows.iloc[targ_cmp]
        gt_row_ind = gt_row.name

        results_dict = {}
        tot_dict = {}
        cons_dict = {}
        vidf_dict = {}
        considered_boxes = []
        all_gt_boxes, all_gt_frames = self.collect_box_frames_from_gt_row(
            gt_row)
        gt_boxes_with_frames = torch.cat(
            [all_gt_boxes, all_gt_frames.unsqueeze(-1)], dim=1)

        gt_frames_all = [
            self.collect_box_frames_from_gt_row(g1)[1]
            for _, g1 in gt_rows.iterrows()
        ]
        # num_srl_args x num_cmp x num_frms x num_prop_per_frm
        pred_boxes_for_verb = pred_row.pred_boxes
        if pred_boxes_for_verb == -1:
            print('oh no')
            return -1

        pred_score_for_verb = pred_row.pred_scores
        pred_cmp_for_verb = pred_row.pred_cmp

        p0_fixed = Counter(
            [x for y in pred_cmp_for_verb for x in y]
        ).most_common()[0][0]

        cmp_msk = pred_row.cmp_msk

        for srl_ind, (
                srl_arg,
                srl_arg_box_indicator,
                srl_arg_box_ind
        ) in enumerate(gt_row.req_cls_pats_mask):

            if srl_arg_box_indicator == 1:
                if gt_row_ind not in results_dict:
                    results_dict[gt_row_ind] = 0
                if gt_row_ind not in tot_dict:
                    tot_dict[gt_row_ind] = 0
                if gt_row_ind not in cons_dict:
                    cons_dict[gt_row_ind] = 0
                if gt_row_ind not in vidf_dict:
                    vidf_dict[gt_row_ind] = 0

                tot_dict[gt_row_ind] += 1
                if srl_ind >= len(pred_boxes_for_verb):
                    continue

                box_inds = torch.tensor(srl_arg_box_ind)
                gt_boxes_frms = torch.index_select(
                    gt_boxes_with_frames, 0, box_inds
                )

                pred_boxes = pred_boxes_for_verb[srl_ind]
                pred_scores = pred_score_for_verb[srl_ind]
                pred_cmp_srl = pred_cmp_for_verb[srl_ind]
                pred_cmp1 = self.pred_cmp_to_pass(p0_fixed, pred_cmp_srl)
                # self.pcs.append(pred_cmp1)
                one_srl_out_dict = self.compute_one_srl(
                    pred_cmp1, pred_boxes, pred_scores,
                    targ_cmp, gt_boxes_frms, gt_frames_all,
                    cmp_msk
                )

                one_srl_out_dict.update({'srl_ind': srl_ind})
                considered_boxes.append(one_srl_out_dict)
                iou = one_srl_out_dict['iou']

                if iou > 0.5:
                    results_dict[gt_row_ind] += 1

        if gt_row_ind in tot_dict:
            cons_out, vidf_out = self.compute_cons_vidf(considered_boxes)

            cons_dict[gt_row_ind] += tot_dict[gt_row_ind] * cons_out
            vidf_dict[gt_row_ind] += tot_dict[gt_row_ind] * vidf_out

        out_dict = {
            'res_dict': results_dict,
            'tot_dict': tot_dict,
            'cons_dict': cons_dict,
            'vidf_dict': vidf_dict,
            'considered_boxes': considered_boxes
        }

        return self.post_proc(out_dict)


class GroundEval_TEMP(GroundEval_SEP):
    def pred_cmp_to_pass(self, p0_cons, p1):
        return p0_cons

    def compute_cons_vidf(self, considered_list):
        """
        considered list: List[Dict] of required stuff
        """
        if len(considered_list) > 0:
            pred_cmps = [c['pred_cmp'] for c in considered_list]

            # self.pcs += pred_cmps
            pred_cmp = Counter(pred_cmps).most_common(1)[0][0]
            targ_cmp = considered_list[0]['targ_cmp']
            cons = all([p == pred_cmp and p >= 0 for p in pred_cmps])
            vid_cor = (pred_cmp == targ_cmp) and cons
            return int(cons), int(vid_cor)
        else:
            return 0, 0

    def compute_one_srl(
            self,
            pred_cmp,
            pred_boxes_for_srl,
            pred_scores_for_srl,
            targ_cmp,
            gt_boxes_with_frames,
            gt_frames_all,
            cmp_msk
    ):
        """
        For sep
        pred_cmp: is the chosen video (1)
        targ_cmp: is the target video (1)
        pred_boxes_for_srl: predicted boxes for
        given srl (#nvids x #nframes x #1-prop(4))
        """
        nvids = len(pred_boxes_for_srl)
        assert len(cmp_msk) == nvids
        # nfrms = len(pred_boxes_for_srl[0])
        # corr_outs = [False for _ in range(nvids)]
        con_outs = {nv: False for nv in range(nvids)}
        # con_boxs = {}
        # con_gts = {}
        # con_frms = {}
        # con_vid = {}
        con_vid = -1
        con_vid_score = 0
        con_vid_scores = {}

        for nv in range(nvids):
            if not cmp_msk[nv] == 1:
                con_outs[nv] = True
                assert [ps0 == 0. for ps0 in pred_scores_for_srl[nv]]
                continue
            pred_boxes = pred_boxes_for_srl[nv]
            pred_scores = pred_scores_for_srl[nv]
            if nv == targ_cmp:
                gt_frms = gt_boxes_with_frames[:, -1].long().tolist()
                assert set(gt_frms).intersection(
                    set(gt_frames_all[nv].tolist())
                ) == set(gt_frms)
                for frm_idx_ind, frm_idx in enumerate(gt_frms):
                    predicted_box = torch.tensor(pred_boxes[frm_idx][:4])
                    pbox = torch.tensor(pred_boxes[frm_idx])
                    groundtruth_box = gt_boxes_with_frames[frm_idx_ind][:4]
                    prediction_score = pred_scores[frm_idx]
                    assert gt_boxes_with_frames[frm_idx_ind][4] == frm_idx
                    iou = box_iou(
                        predicted_box.float(),
                        groundtruth_box.float()
                    )
                    # TODO: Check why prediction scores
                    # are ridiculously low!!
                    if iou > 0.5 and prediction_score > self.prob_thresh:
                        con_iou = iou
                        con_box = predicted_box
                        con_box_full = pbox
                        con_gt = groundtruth_box
                        con_frm = frm_idx
                        con_vid_score = prediction_score
                        con_outs[nv] = True
                        con_vid = nv
            else:
                gt_frms = gt_frames_all[nv]
                # rfrms = [i for i in range(
                corr = True
                for frm_idx_ind, frm_idx in enumerate(gt_frms):
                    prediction_score = pred_scores[frm_idx]
                    if prediction_score > self.prob_thresh:
                        corr = False
                        break
                con_outs[nv] = corr
                if not corr:
                    con_vid = nv
                    con_vid_scores[nv] = prediction_score

        if all(list(con_outs.values())):
            return {
                'targ_cmp': targ_cmp,
                'pred_cmp': con_vid,
                'pred_score': con_vid_score,
                'predicted_box': con_box,
                'pbox': con_box_full,
                'gt_box': con_gt,
                'frm_idx': con_frm,
                'iou': con_iou
            }

        con_vid_list = sorted(
            [(k, v) for k, v in con_vid_scores.items()],
            key=lambda x: x[-1], reverse=True
        )
        if len(con_vid_list) > 0:
            con_vid = con_vid_list[0][0]
            con_vid_score = con_vid_list[0][1]
        else:
            con_vid = -1
            con_vid_score = 0
        return {
            'targ_cmp': targ_cmp,
            'pred_cmp': con_vid,
            'pred_score': con_vid_score,
            'iou': torch.tensor(0)
        }


class GroundEval_SPAT(GroundEval_SEP):
    def pred_cmp_to_pass(self, p0_cons, p1):
        return p1

    def compute_cons_vidf(self, considered_list):
        """
        considered list: List[Dict] of required stuff
        """
        if len(considered_list) > 0:
            pred_cmps = [c['pred_cmp'] for c in considered_list]
            pscores = [c['pred_score'] for c in considered_list]
            # self.pcs += pred_cmps
            pred_cmp = Counter(pred_cmps).most_common(1)[0][0]
            targ_cmp = considered_list[0]['targ_cmp']
            # cons = np.mean([p == pred_cmp and p >= 0 for p in pred_cmps])
            cons = all([p == pred_cmp for p in pred_cmps])
            # cons = all([p == pred_cmp for p, score in zip(pred_cmps, pscores)])
            # vid_cor = np.mean([p == targ_cmp and p >= 0 for p in pred_cmps])
            vid_cor = all([p == targ_cmp and p >= 0 for p in pred_cmps])
            # c0 = considered_list[0]
            # cons = torch.mean(
            # vid_cor = c0['pred_cmp'] == c0['targ_cmp']
            # self.stuff += [0]
            return int(cons), int(vid_cor)
        else:
            # self.stuff += [1]
            return 0, 0

    def compute_one_srl(
            self,
            pred_cmp_for_srl,
            pred_boxes_for_srl,
            pred_scores_for_srl,
            targ_cmp,
            gt_boxes_with_frames,
            gt_frames_all,
            cmp_msk
    ):
        """
        For spatial
        targ_cmp: is the target video (1)
        pred_boxes_for_srl: predicted boxes for
        given srl (#nvids x #1-prop(4))
        """
        nfrms = len(pred_boxes_for_srl[0])
        con_vid = -1
        con_vid_score = 0
        con_vid_scores = {}
        con_vid_boxes = {}
        # req_frms = list(set([frm.tolist() for frm1 in gt_frames_all
        # for frm in frm1]))

        req_frms = [i for i in range(nfrms)]

        con_outs = {nv: False for nv in req_frms}

        gt_frms = set(gt_boxes_with_frames[:, -1].long().tolist())
        assert gt_frms.intersection(
            set(gt_frames_all[targ_cmp].tolist())
        ) == gt_frms

        delta = torch.zeros(gt_boxes_with_frames.size(1)).long()
        delta[[0, 2]] = 720

        gt_box_for_frms = {}
        for g in gt_boxes_with_frames:
            gfrm = g[4].item()
            if gfrm not in gt_box_for_frms:
                gt_box_for_frms[gfrm] = []
            gt_box_for_frms[gfrm].append(g + delta * targ_cmp)

        for nf in req_frms:
            nv = pred_cmp_for_srl[nf]
            assert cmp_msk[nv] == 1
            prediction_score = pred_scores_for_srl[nv][nf]
            pred_boxes = pred_boxes_for_srl[nv][nf]
            if nf in gt_frms:
                if nv == targ_cmp:
                    predicted_box = torch.tensor(
                        pred_boxes[:4]
                    )
                    pbox = torch.tensor(
                        pred_boxes
                    )

                    assert nf in gt_box_for_frms

                    groundtruth_boxes = gt_box_for_frms[nf]
                    for groundtruth_box in groundtruth_boxes:
                        assert groundtruth_box[4] == nf
                        iou = box_iou(
                            predicted_box.float(),
                            groundtruth_box[:4].float()
                        )
                        # TODO: Check why prediction scores
                        # are ridiculously low!!
                        if iou > 0.5 and prediction_score > self.prob_thresh:
                            con_iou = iou
                            con_box = predicted_box
                            con_box_full = pbox
                            con_gt = groundtruth_box
                            con_frm = nf
                            con_vid_score = prediction_score
                            con_outs[nf] = True
                            con_vid = nv
                else:
                    corr = False
                    con_outs[nf] = corr
            else:
                corr = True
                if nv != targ_cmp and prediction_score > self.prob_thresh:
                    corr = False
                con_outs[nf] = corr
                # if not corr:
                con_vid_scores[nf] = prediction_score
                con_vid_boxes[nf] = pred_boxes

        if all(list(con_outs.values())):
            return {
                'targ_cmp': targ_cmp,
                'pred_cmp': con_vid,
                'pred_score': con_vid_score,
                'predicted_box': con_box,
                'pbox': con_box_full,
                'gt_box': con_gt,
                'frm_idx': con_frm,
                'iou': con_iou
            }

        con_vid_list = sorted(
            [(k, v, con_vid_boxes[k]) for k, v in con_vid_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        if len(con_vid_list) > 0:
            con_vid = -con_vid_list[0][0]
            con_vid_score = con_vid_list[0][1]
            con_vid_box = torch.tensor(con_vid_list[0][2])
        else:
            con_vid = -5
            con_vid_score = 0
            con_vid_box = torch.tensor([0, 0, 0, 0, 0])
        return {
            'targ_cmp': targ_cmp,
            'pred_cmp': con_vid,
            'pred_score': con_vid_score,
            'predicted_box': con_vid_box,
            'gt_box': gt_box_for_frms,
            'iou': torch.tensor(0)
        }


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
        grnd_eval = GroundEval_SEP(cfg, comm)
    elif conc_type == 'temp':
        grnd_eval = GroundEval_TEMP(cfg, comm)
    elif conc_type == 'spat':
        grnd_eval = GroundEval_SPAT(cfg, comm)
    else:
        raise NotImplementedError
    out = grnd_eval.eval_ground_acc(pred_file, split_type=split_type)
    # to_print = ['avg1', 'avg2']
    # print(Counter(grnd_eval.pcs))
    met_keys = ['avg1', 'avg1_cons',
                'avg1_vidf', 'avg1_strict']
    print({k: out[k] for k in met_keys})
    # print(Counter(grnd_eval.stuff))
    # return out
    return


if __name__ == '__main__':
    fire.Fire(main)
