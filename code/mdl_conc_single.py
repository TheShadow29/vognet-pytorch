"""
Concatenate to a Single Video
"""
import torch
from torch import nn
from torch.nn import functional as F
from box_utils import bbox_overlaps


class ConcBase(nn.Module):
    """
    Base Model for concatenation
    """

    def set_args_conc(self):
        """
        Conc Type specific args
        """
        return


class ConcTEMP(ConcBase):
    def conc_encode(self, conc_feats, inp):
        ncmp = inp['new_srl_idxs'].size(1)
        nfrm = ncmp * self.num_sampled_frm
        nppf = self.num_prop_per_frm
        return self.conc_encode_item(conc_feats, inp, nfrm, nppf, 1)

    def simple_obj_interact_input(self, prop_seg_feats, inp):
        # B, num_cmp, num_props, psdim = prop_seg_feats.shape
        num_cmp = inp['new_srl_idxs'].size(1)
        return self.simple_obj_interact(
            prop_seg_feats, inp, 1,
            num_cmp * self.num_sampled_frm,
            self.num_prop_per_frm
        )

    def get_num_cmp_msk(self, inp, out_shape):
        num_cmp = inp['new_srl_idxs'].size(1)
        B, num_verbs, num_srl_args, seq_len = inp['srl_arg_words_ind'].shape
        num_cmp_msk = inp['num_cmp_msk'].view(
            B, 1, 1, num_cmp, 1
        ).expand(
            B, num_verbs, num_srl_args, num_cmp,
            self.num_sampled_frm * self.num_prop_per_frm
        ).contiguous(
        ).view(*out_shape)
        return num_cmp_msk

    def concat_prop_seg_feats(self, prop_feats, seg_feats, inp):
        B, num_v_frms, sdim = seg_feats.shape
        num_cmp = inp['new_srl_idxs'].size(1)

        prop_seg_feats = torch.cat(
            [prop_feats.view(
                B, 1, num_cmp * self.num_sampled_frm,
                self.num_prop_per_frm, prop_feats.size(-1)),
                seg_feats.view(B, 1, num_v_frms, 1, sdim).expand(
             B, 1, num_cmp * self.num_sampled_frm,
             self.num_prop_per_frm, sdim)
             ], dim=-1).view(
            B, 1, num_cmp * self.num_sampled_frm * self.num_prop_per_frm,
            prop_feats.size(-1) + seg_feats.size(-1)
        )
        return prop_seg_feats

    def forward(self, inp):
        """
        Main difference is that prop feats/seg features
        have an extra dimension
        """
        # B x 6 x 5 x 40
        # 6 is num_cmp for a sent
        # 5 is num args in a sent
        # 40 is seq length for each arg
        B, num_verbs, num_srl_args, seq_len = inp['srl_arg_words_ind'].shape
        # B*num_cmp x seq_len
        src_toks = self.get_srl_arg_seq_to_sent_seq(inp)
        # B*num_cmp x seq_len
        src_lens = inp['srl_arg_word_mask_len'].view(B*num_verbs, -1)
        # B*num_cmp x seq_len x 256
        lstm_outs = self.lang_encode(src_toks, src_lens)
        lstm_encoded = lstm_outs['lstm_full_output']

        # B x 1 x 5 x 512
        srl_arg_lstm_encoded = self.retrieve_srl_arg_from_lang_encode(
            lstm_encoded, inp
        )

        # Get visual features
        # B x 40*100 x 512
        prop_feats = self.prop_feats_encode(inp)

        # Get seg features
        # B x 40 x 512
        seg_feats = self.seg_feats_encode(inp)
        B, num_v_frms, sdim = seg_feats.shape
        # Simple conc seg_feats
        prop_seg_feats = self.concat_prop_seg_feats(
            prop_feats, seg_feats, inp
        )

        # Object Interaction if to be done
        prop_seg_feats = self.simple_obj_interact_input(
            prop_seg_feats, inp
        )

        # B x 1 x num_srl_args x 4*num_props x vf+lf dim
        conc_feats = self.concate_vis_lang_feats(
            prop_seg_feats, srl_arg_lstm_encoded
        )

        # B x num_cmp x num_srl_args x 4*num_props x vf+lf dim
        conc_feats_out_dict = self.conc_encode(conc_feats, inp)
        conc_feats_out = conc_feats_out_dict['conc_feats_out']

        num_cmp_msk = self.get_num_cmp_msk(inp, conc_feats_out.shape)
        srl_ind_msk = inp['srl_arg_inds_msk'].unsqueeze(-1).expand(
            *conc_feats_out.shape)
        conc_feats_out_eval = torch.sigmoid(
            conc_feats_out) * srl_ind_msk.float() * num_cmp_msk.float()

        return {
            'mdl_outs': conc_feats_out,
            'mdl_outs_eval': conc_feats_out_eval,
        }


class ConcSPAT(ConcTEMP):
    def conc_encode(self, conc_feats, inp):
        ncmp = inp['new_srl_idxs'].size(1)
        nfrm = self.num_sampled_frm
        nppf = ncmp * self.num_prop_per_frm
        return self.conc_encode_item(conc_feats, inp, nfrm, nppf, 1)

    def simple_obj_interact_input(self, prop_seg_feats, inp):
        num_cmp = inp['new_srl_idxs'].size(1)
        return self.simple_obj_interact(
            prop_seg_feats, inp, 1,
            self.num_sampled_frm, num_cmp * self.num_prop_per_frm
        )

    def get_num_cmp_msk(self, inp, out_shape):
        num_cmp = inp['new_srl_idxs'].size(1)
        B, num_verbs, num_srl_args, seq_len = inp['srl_arg_words_ind'].shape
        num_cmp_msk = inp['num_cmp_msk'].view(
            B, 1, 1, 1, num_cmp, 1
        ).expand(
            B, num_verbs, num_srl_args, self.num_sampled_frm,
            num_cmp, self.num_prop_per_frm
        ).contiguous(
        ).view(*out_shape)
        return num_cmp_msk

    def concat_prop_seg_feats(self, prop_feats, seg_feats, inp):
        B, num_v_frms, sdim = seg_feats.shape
        num_cmp = inp['new_srl_idxs'].size(1)
        prop_seg_feats = torch.cat(
            [
                prop_feats.view(
                    B, 1, self.num_sampled_frm * num_cmp,
                    self.num_prop_per_frm, prop_feats.size(-1)
                ), seg_feats.view(B, 1, num_v_frms, 1, sdim).expand(
                    B, 1, self.num_sampled_frm * num_cmp,
                    self.num_prop_per_frm, sdim
                )
            ],
            dim=-1
        ).view(
            B, 1, self.num_sampled_frm * num_cmp * self.num_prop_per_frm,
            prop_feats.size(-1) + seg_feats.size(-1)
        )
        return prop_seg_feats

    def forward(self, inp):
        return ConcTEMP.forward(self, inp)


class LossB_TEMP(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.loss_keys = ['loss', 'mdl_out_loss']
        self.loss_lambda = self.cfg.loss.loss_lambda
        self.after_init()

    def after_init(self):
        pass

    def get_targets_from_overlaps(self, overlaps, inp):
        """
        Use the given overlaps to produce the targets
        overlaps: B x num_cmp x 1000 x 100
        """
        # to_consider = overlaps > 0.5
        targets = overlaps

        srl_boxes = inp['srl_boxes']
        # B, num_cmp, num_srl_args, num_box_per_srl = srl_boxes.shape
        B, num_verbs, num_srl_args, num_box_per_srl = srl_boxes.shape
        B, num_props, num_gt_box = targets.shape

        srl_boxes_reshaped = srl_boxes.view(
            B, num_verbs, num_srl_args, 1, num_box_per_srl).expand(
                B, num_verbs, num_srl_args, num_props, num_box_per_srl)

        targets_reshaped = targets.view(
            B, 1, 1, num_props, num_gt_box).expand(
                B, num_verbs, num_srl_args, num_props, num_gt_box)

        # Choose only those proposals which are ground-truth
        # for given srl
        targets_to_use = torch.gather(
            targets_reshaped, dim=-1, index=srl_boxes_reshaped)

        srl_boxes_lens = inp['srl_boxes_lens']
        targets_to_use = (
            targets_to_use * srl_boxes_lens.float().unsqueeze(
                -2).expand(*targets_to_use.shape)
        )

        targets_to_use = targets_to_use.max(dim=-1)[0] > 0.5

        return targets_to_use

    def compute_overlaps(self, inp):

        pad_props = inp['pad_proposals']
        gt_bboxs = inp['pad_gt_bboxs']
        frm_msk = inp['pad_frm_mask']
        pnt_msk = inp['pad_pnt_mask']

        try:
            overlaps = bbox_overlaps(
                pad_props, gt_bboxs,
                (frm_msk | pnt_msk.unsqueeze(-1))
            )
        except:
            import pdb
            pdb.set_trace()
            overlaps = bbox_overlaps(
                pad_props, gt_bboxs,
                (frm_msk | pnt_msk.unsqueeze(-1)))

        return overlaps

    def compute_loss_targets(self, inp):
        """
        Compute the targets, based on iou
        overlaps
        """
        num_cmp = inp['new_srl_idxs'].size(1)
        overlaps = self.compute_overlaps(inp)
        B, num_tot_props, num_gt = overlaps.shape
        assert num_tot_props % num_cmp == 0
        num_props = num_tot_props // num_cmp
        overlaps_msk = overlaps.new_zeros(B, num_cmp, num_props, num_gt)

        targ_cmp = inp['target_cmp']

        overlaps_msk.scatter_(
            dim=1,
            index=targ_cmp.view(B, 1, 1, 1).expand(
                B, num_cmp, num_props, num_gt),
            src=overlaps_msk.new_ones(*overlaps_msk.shape)
        )

        overlaps_msk = overlaps_msk.view(B, num_tot_props, num_gt)
        overlaps_one_targ = overlaps * overlaps_msk
        targets_one = self.get_targets_from_overlaps(overlaps_one_targ, inp)
        return {
            'targets_one': targets_one,
        }

    def compute_mdl_loss(self, mdl_outs, targets_one, inp):
        weights = None
        tot_loss = F.binary_cross_entropy_with_logits(
            mdl_outs, target=targets_one.float(),
            weight=weights,
            reduction='none'
        )

        num_cmp_msk = inp['num_cmp_msk']
        B, num_cmp = num_cmp_msk.shape

        srl_arg_boxes_mask = inp['srl_arg_boxes_mask']
        B, num_verbs, num_srl_args = srl_arg_boxes_mask.shape

        boxes_msk = (
            srl_arg_boxes_mask.view(
                B, num_verbs, num_srl_args, 1).expand(
                    B, num_verbs, num_srl_args, num_cmp).float() *
            num_cmp_msk.view(
                B, 1, 1, num_cmp).expand(
                    B, num_verbs, num_srl_args, num_cmp).float()
        )
        num_props_per_vid = targets_one.size(-1) // num_cmp
        # B x num_cmp x num_srl_args -> B x num_cmp x num_srl x 4000
        boxes_msk = boxes_msk.unsqueeze(
            -1).expand(
                B, num_verbs, num_srl_args, num_cmp, num_props_per_vid
        ).contiguous().view(
            B, num_verbs, num_srl_args, num_cmp * num_props_per_vid)

        multiplier = tot_loss.size(-1)
        if srl_arg_boxes_mask.max() > 0:
            out_loss = torch.masked_select(tot_loss, boxes_msk.byte())
        else:
            # TODO: NEED TO check what is wrong here
            out_loss = tot_loss
        mdl_out_loss = out_loss.mean() * multiplier
        # mdl_out_loss = out_loss * 1000
        return mdl_out_loss

    def forward(self, out, inp):
        targets_all = self.compute_loss_targets(inp)
        targets_n = targets_all['targets_one']

        mdl_outs = out['mdl_outs']

        mdl_out_loss = self.compute_mdl_loss(mdl_outs, targets_n, inp)

        out_loss = mdl_out_loss

        out_loss_dict = {
            'loss': out_loss,
            'mdl_out_loss': mdl_out_loss,
        }

        return {k: v * self.loss_lambda for k, v in out_loss_dict.items()}


class LossB_SPAT(LossB_TEMP):
    def after_init(self):
        self.loss_keys = ['loss', 'mdl_out_loss']

        self.num_sampled_frm = self.cfg.ds.num_sampled_frm
        self.num_prop_per_frm = self.comm.num_prop_per_frm

    def compute_loss_targets(self, inp):
        """
        Compute the targets, based on iou
        overlaps
        """
        num_cmp = inp['new_srl_idxs'].size(1)
        overlaps = self.compute_overlaps(inp)
        B, num_tot_props, num_gt = overlaps.shape
        assert num_tot_props % num_cmp == 0

        overlaps_msk = overlaps.new_zeros(
            B, self.num_sampled_frm, num_cmp,
            self.num_prop_per_frm, num_gt
        )

        targ_cmp = inp['target_cmp']
        overlaps_msk.scatter_(
            dim=2,
            index=targ_cmp.view(B, 1, 1, 1, 1).expand(
                B, self.num_sampled_frm, num_cmp, self.num_prop_per_frm, num_gt
            ),
            src=overlaps_msk.new_ones(*overlaps_msk.shape)
        )

        overlaps_msk = overlaps_msk.view(B, num_tot_props, num_gt)
        overlaps_one_targ = overlaps * overlaps_msk
        targets_one = self.get_targets_from_overlaps(overlaps_one_targ, inp)
        return {
            'targets_one': targets_one,
        }

    def compute_mdl_loss(self, mdl_outs, targets_one, inp):
        weights = None
        tot_loss = F.binary_cross_entropy_with_logits(
            mdl_outs, target=targets_one.float(),
            weight=weights,
            reduction='none'
        )

        num_cmp_msk = inp['num_cmp_msk']
        B, num_cmp = num_cmp_msk.shape

        srl_arg_boxes_mask = inp['srl_arg_boxes_mask']

        B, num_verbs, num_srl_args = srl_arg_boxes_mask.shape

        boxes_msk = (
            srl_arg_boxes_mask.view(
                B, num_verbs, num_srl_args, 1).expand(
                    B, num_verbs, num_srl_args, num_cmp).float() *
            num_cmp_msk.view(
                B, 1, 1, num_cmp).expand(
                    B, num_verbs, num_srl_args, num_cmp).float()
        )

        num_tot_props = targets_one.size(-1)
        # B x num_cmp x num_srl_args -> B x num_cmp x num_srl x 4000
        boxes_msk = boxes_msk.view(
            B, num_verbs, num_srl_args, 1, num_cmp, 1
        ).expand(
            B, num_verbs, num_srl_args, self.num_sampled_frm,
            num_cmp, self.num_prop_per_frm
        ).contiguous().view(
            B, num_verbs, num_srl_args, num_tot_props
        )

        multiplier = tot_loss.size(-1)
        if srl_arg_boxes_mask.max() > 0:
            out_loss = torch.masked_select(tot_loss, boxes_msk.byte())
        else:
            # TODO: NEED TO check what is wrong here
            out_loss = tot_loss
        mdl_out_loss = out_loss.mean() * multiplier

        return mdl_out_loss

    def forward(self, out, inp):
        targets_all = self.compute_loss_targets(inp)
        targets_n = targets_all['targets_one']

        mdl_outs = out['mdl_outs']

        mdl_out_loss = self.compute_mdl_loss(mdl_outs, targets_n, inp)

        out_loss = mdl_out_loss

        out_loss_dict = {
            'loss': out_loss,
            'mdl_out_loss': mdl_out_loss,
        }

        return {k: v * self.loss_lambda for k, v in out_loss_dict.items()}
