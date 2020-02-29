"""
Concatenate to a Single Video
"""
import torch
from torch.nn import functional as F
from mdl_base import AnetSimpleBCEMdlDS4, ConcBase
from mdl_conc import LossB_DS4
from box_utils import bbox_overlaps


class ConcTemp(ConcBase):
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
        num_cmp = inp['new_srl_idxs'].size(1)
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

        verb_feats = lstm_outs['final_hidden']
        B_num_cmp, ldim = verb_feats.shape
        # if B_num_cmp == B:
        verb_feats = verb_feats.view(B, 1, ldim).expand(
            B, num_cmp, ldim)
        # else:
        # verb_feats = verb_feats.view(B, num_cmp, ldim)

        # seg_feats_for_verb = seg_feats.view(
        #     B, num_cmp, self.num_sampled_frm, sdim).mean(dim=-2).view(
        #         B, num_cmp, sdim
        # )

        # # B x num_cmp
        # vidf_outs = self.compute_seg_verb_feats_out(
        #     seg_feats_for_verb, verb_feats
        # )

        # # # B x num_cmp x num_srl_args
        # prop_scores_max_boxes, _ = torch.max(
        #     torch.sigmoid(conc_feats_out), dim=-1)

        # # B x num_cmp
        # fin_scores = prop_scores_max_boxes.sum(dim=-1)
        num_cmp_msk = self.get_num_cmp_msk(inp, conc_feats_out.shape)
        srl_ind_msk = inp['srl_arg_inds_msk'].unsqueeze(-1).expand(
            *conc_feats_out.shape)
        conc_feats_out_eval = torch.sigmoid(
            conc_feats_out) * srl_ind_msk.float() * num_cmp_msk.float()

        return {
            'mdl_outs': conc_feats_out,
            # 'vidf_outs': vidf_outs,
            'mdl_outs_eval': conc_feats_out_eval,
        }


class ConcSPAT(ConcTemp):
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
        return ConcTemp.forward(self, inp)


class LossB_SSJ1_Temporal_DS4(LossB_DS4):
    def after_init(self):
        self.loss_keys = ['loss', 'mdl_out_loss']

    def compute_overlaps(self, inp):

        pad_props = inp['pad_proposals']
        gt_bboxs = inp['pad_gt_bboxs']
        frm_msk = inp['pad_frm_mask']
        pnt_msk = inp['pad_pnt_mask']
        # num_cmp = inp['new_srl_idxs'].size(1)
        # num_cmp_t = inp['num_cmp'][0].item()
        num_cmp_t = 1
        try:
            B = pad_props.size(0)
            overlaps = bbox_overlaps(
                pad_props, gt_bboxs,
                (frm_msk | pnt_msk.unsqueeze(-1))
            )
            # overlaps = overlaps.view(B, num_cmp_t, *overlaps.shape[1:])

        except:
            import pdb
            pdb.set_trace()
            overlaps = bbox_overlaps(
                pad_props, gt_bboxs,
                (frm_msk | pnt_msk.unsqueeze(-1)))

        return overlaps

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
        # targ_cmp = inp['target_cmp'][0].item()
        targ_cmp = inp['target_cmp']
        # overlaps_msk[:, targ_cmp, ...] = 1
        overlaps_msk.scatter_(
            dim=1,
            index=targ_cmp.view(B, 1, 1, 1).expand(
                B, num_cmp, num_props, num_gt),
            src=overlaps_msk.new_ones(*overlaps_msk.shape)
        )

        # overlaps_msk[:, 0, ...] = 1
        overlaps_msk = overlaps_msk.view(B, num_tot_props, num_gt)
        overlaps_one_targ = overlaps * overlaps_msk
        targets_one = self.get_targets_from_overlaps(overlaps_one_targ, inp)
        # targets_all = self.get_targets_from_overlaps(overlaps, inp)
        return {
            'targets_one': targets_one,
            # 'targets_all': targets_all
        }

    def compute_mdl_loss(self, mdl_outs, targets_one, inp):
        # B x num_cmp x num_srl_args x  num_props
        # ps = torch.sigmoid(mdl_outs)
        # enc_tgt = targets_one.float()
        # weights = enc_tgt * (1 - ps) + (1 - enc_tgt) * ps
        # alpha = 0.25
        # gamma = 2
        # alphas = ((1 - enc_tgt) * alpha + enc_tgt * (1 - alpha))
        # weights.pow_(gamma).mul_(alphas)
        # weights = weights.detach()
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
        # if num_verbs != num_cmp:
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
            # TODO: Check if mean is correct thing to do here.
            # out_loss = torch.masked_select(tot_loss, srl_arg_boxes_mask.byte())
            out_loss = torch.masked_select(tot_loss, boxes_msk.byte())
            # out_loss = torch.masked_select(tot_loss, srl2.byte())

        else:
            # TODO: NEED TO check what is wrong here
            out_loss = tot_loss
        mdl_out_loss = out_loss.mean() * multiplier
        # mdl_out_loss = out_loss * 1000
        return mdl_out_loss

    def compute_vidf_loss_simple(self, vidf_outs, inp):
        """
        vidf_outs are fin scores: B x ncmp x nfrms
        """
        B, ncmp, nfrm = vidf_outs.shape
        targs = vidf_outs.new_zeros(*vidf_outs.shape)
        # targ_cmp = inp['target_cmp'][0].item()
        targ_cmp = inp['target_cmp']

        targs.scatter_(
            dim=1,
            index=targ_cmp.view(B, 1, 1).expand(B, ncmp, nfrm),
            src=targs.new_ones(*targs.shape)
        )
        # targs[:, targ_cmp] = 1

        # B x ncmp x nfrms
        out_loss = F.binary_cross_entropy(vidf_outs, targs, reduction='none')
        # mult = out_loss.size(-1)
        mult = 1/nfrm
        # B x ncmp
        msk = inp['num_cmp_msk']  #
        out_loss = torch.masked_select(out_loss.sum(dim=-1) * msk.float(),
                                       msk.byte()) * mult
        return out_loss.mean()

    def forward(self, out, inp):
        targets_all = self.compute_loss_targets(inp)
        # targets_one = targets_all['targets_one']
        # targets_one = targets_all['targets_one']
        # targets_n = targets_all['targets_all']
        targets_n = targets_all['targets_one']
        # import pdb
        # pdb.set_trace()
        mdl_outs = out['mdl_outs']
        # mdl_out_loss = self.compute_mdl_loss(mdl_outs, targets_one, inp)
        mdl_out_loss = self.compute_mdl_loss(mdl_outs, targets_n, inp)

        verb_outs = out['vidf_outs']

        verb_loss = F.binary_cross_entropy_with_logits(
            verb_outs,
            inp['verb_cmp'].float(),
            reduction='none'
        )

        vcc_msk = inp['verb_cross_cmp_msk'].float()
        vcc_msk = (vcc_msk.sum(dim=-1) > 0).float()
        verb_loss = verb_loss * vcc_msk
        verb_loss = torch.masked_select(
            verb_loss, vcc_msk.byte()).mean()

        # out_loss = mdl_out_loss + verb_loss
        out_loss = mdl_out_loss

        # vidf_loss = mdl_out_loss.new_zeros(mdl_out_loss.shape)
        # if not self.cfg.mdl.loss.only_vid_loss:
        # out_loss = mdl_out_loss + verb_loss
        # out_loss = mdl_out_loss + verb_loss
        # else:
        # out_loss = vidf_loss
        out_loss_dict = {
            'loss': out_loss,
            'mdl_out_loss': mdl_out_loss,
            # 'vidf_loss': vidf_loss,
            # 'verb_loss': verb_loss
        }

        return {k: v * self.loss_lambda for k, v in out_loss_dict.items()}


class LossB_SSJ1_Spatial_DS4(LossB_SSJ1_Temporal_DS4):
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
        # num_props = num_tot_props // num_cmp

        overlaps_msk = overlaps.new_zeros(
            B, self.num_sampled_frm, num_cmp,
            self.num_prop_per_frm, num_gt
        )

        targ_cmp = inp['target_cmp']
        # overlaps_msk[:, targ_cmp, ...] = 1
        overlaps_msk.scatter_(
            dim=2,
            index=targ_cmp.view(B, 1, 1, 1, 1).expand(
                B, self.num_sampled_frm, num_cmp, self.num_prop_per_frm, num_gt
            ),
            src=overlaps_msk.new_ones(*overlaps_msk.shape)
        )

        # overlaps_msk[:, :, 0, ...] = 1
        overlaps_msk = overlaps_msk.view(B, num_tot_props, num_gt)
        overlaps_one_targ = overlaps * overlaps_msk
        targets_one = self.get_targets_from_overlaps(overlaps_one_targ, inp)
        # targets_all = self.get_targets_from_overlaps(overlaps, inp)
        return {
            'targets_one': targets_one,
            # 'targets_all': targets_all
        }

    def compute_mdl_loss(self, mdl_outs, targets_one, inp):
        # B x num_cmp x num_srl_args x  num_props
        # ps = torch.sigmoid(mdl_outs)
        # enc_tgt = targets_one.float()
        # weights = enc_tgt * (1 - ps) + (1 - enc_tgt) * ps
        # alpha = 0.25
        # gamma = 2
        # alphas = ((1 - enc_tgt) * alpha + enc_tgt * (1 - alpha))
        # weights.pow_(gamma).mul_(alphas)
        # weights = weights.detach()
        weights = None
        tot_loss = F.binary_cross_entropy_with_logits(
            mdl_outs, target=targets_one.float(),
            weight=weights,
            reduction='none'
        )

        num_cmp_msk = inp['num_cmp_msk']
        B, num_cmp = num_cmp_msk.shape

        srl_arg_boxes_mask = inp['srl_arg_boxes_mask']
        # srl_arg_boxes_mask[...] = 1
        B, num_verbs, num_srl_args = srl_arg_boxes_mask.shape
        # if num_verbs != num_cmp:
        boxes_msk = (
            srl_arg_boxes_mask.view(
                B, num_verbs, num_srl_args, 1).expand(
                    B, num_verbs, num_srl_args, num_cmp).float() *
            num_cmp_msk.view(
                B, 1, 1, num_cmp).expand(
                    B, num_verbs, num_srl_args, num_cmp).float()
        )

        num_tot_props = targets_one.size(-1)
        # num_props_per_vid = num_tot_props // num_cmp
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
            # TODO: Check if mean is correct thing to do here.
            # out_loss = torch.masked_select(tot_loss, srl_arg_boxes_mask.byte())
            out_loss = torch.masked_select(tot_loss, boxes_msk.byte())
            # out_loss = torch.masked_select(tot_loss, srl2.byte())

        else:
            # TODO: NEED TO check what is wrong here
            out_loss = tot_loss
        mdl_out_loss = out_loss.mean() * multiplier
        # mdl_out_loss = out_loss * 1000
        return mdl_out_loss

    def forward(self, out, inp):
        targets_all = self.compute_loss_targets(inp)
        # targets_one = targets_all['targets_one']
        # targets_one = targets_all['targets_one']
        # targets_n = targets_all['targets_all']
        targets_n = targets_all['targets_one']
        # import pdb
        # pdb.set_trace()
        mdl_outs = out['mdl_outs']
        # mdl_out_loss = self.compute_mdl_loss(mdl_outs, targets_one, inp)
        mdl_out_loss = self.compute_mdl_loss(mdl_outs, targets_n, inp)

        # verb_outs = out['vidf_outs']

        # verb_loss = F.binary_cross_entropy_with_logits(
        #     verb_outs,
        #     inp['verb_cmp'].float(),
        #     reduction='none'
        # )
        # vcc_msk = inp['verb_cross_cmp_msk'].float()
        # vcc_msk = (vcc_msk.sum(dim=-1) > 0).float()
        # verb_loss = verb_loss * vcc_msk
        # verb_loss = torch.masked_select(
        # verb_loss, vcc_msk.byte()
        # ).mean()

        # vidf_loss = mdl_out_loss.new_zeros(mdl_out_loss.shape)
        # if not self.cfg.mdl.loss.only_vid_loss:
        # out_loss = mdl_out_loss + verb_loss
        out_loss = mdl_out_loss
        # out_loss = mdl_out_loss + verb_loss
        # else:
        # out_loss = vidf_loss
        out_loss_dict = {
            'loss': out_loss,
            'mdl_out_loss': mdl_out_loss,
            # 'vidf_loss': vidf_loss,
            # 'verb_loss': verb_loss
        }

        return {k: v * self.loss_lambda for k, v in out_loss_dict.items()}
