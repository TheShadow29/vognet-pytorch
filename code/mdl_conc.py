"""
Take care of SEP case.

"""
from mdl_base import AnetSimpleBCEMdlDS4, ConcBase

import torch
from torch import nn
from dat_loader import get_data
from torch.nn import functional as F
# from transformer import Transformer
# from relative_transformer import RelTransformer
from mdl_srl_utils import do_cross, combine_first_ax
from box_utils import bbox_overlaps


class ConcSEP(ConcBase):
    def conc_encode(self, conc_feats, inp):
        nfrm = self.num_sampled_frm
        nppf = self.num_prop_per_frm
        ncmp = inp['new_srl_idxs'].size(1)
        return self.conc_encode_item(conc_feats, inp, nfrm, nppf, ncmp)

    def simple_obj_interact_input(self, prop_seg_feats, inp):
        B, num_cmp, num_props, psdim = prop_seg_feats.shape
        return self.simple_obj_interact(
            prop_seg_feats, inp,
            num_cmp, self.num_sampled_frm,
            self.num_prop_per_frm
        )

    def set_args_conc(self):
        self.nfrms = self.num_sampled_frm
        self.nppf = self.num_prop_per_frm

    def concat_prop_seg_feats(self, prop_feats, seg_feats, inp):
        B, num_cmp, num_props, pdim = prop_feats.shape
        # Simple conc seg_feats
        prop_seg_feats = torch.cat(
            [
                prop_feats.view(
                    B, num_cmp, self.num_sampled_frm,
                    self.num_prop_per_frm, prop_feats.size(-1)
                ),
                seg_feats.unsqueeze(-2).expand(
                    B, num_cmp, self.num_sampled_frm,
                    self.num_prop_per_frm, seg_feats.size(-1)
                )
            ], dim=-1
        ).view(
            B, num_cmp, self.num_sampled_frm*self.num_prop_per_frm,
            prop_feats.size(-1) + seg_feats.size(-1)
        )
        # B x num_cmp x nfrm*nppf x psdim
        return prop_seg_feats

    def compute_fin_scores(self, conc_out_dict, inp, vidf_outs=None):
        """
        output fin scores should be of shape
        B x num_cmp
        prop_scores: B x num_cmp x num_srl_args x num_props
        """
        prop_scores1 = conc_out_dict['conc_feats_out'].clone().detach()
        prop_scores = torch.sigmoid(prop_scores1)
        # prop_scores = prop_scores1
        if self.cfg.mdl.use_vis_msk:
            # B x num_cmp x num_srl_args
            prop_scores_max_boxes, _ = torch.max(prop_scores, dim=-1)

            # B x num_cmp x num_srl_args
            srl_arg_inds_msk = inp['srl_arg_inds_msk'].float()
            B, num_verbs, num_srl_args = srl_arg_inds_msk.shape

            num_cmp = prop_scores.size(1)

            if vidf_outs is not None:
                # add vidf outs to the verb places
                vidf_outs = torch.sigmoid(vidf_outs)

                # B x num_cmp -> B x num_cmp x num_srl_args
                vidf_outs = vidf_outs.unsqueeze(-1).expand(
                    *prop_scores_max_boxes.shape
                )
                vmsk = inp['verb_ind_in_srl']

                if vmsk.size(1) == 1 and num_cmp > 1:
                    vmsk = vmsk.expand(-1, num_cmp)
                # B x num_cmp
                vmsk = vmsk.view(
                    B, num_cmp, 1).expand(
                        B, num_cmp, num_srl_args
                )
                prop_scores_max_boxes.scatter_(
                    dim=2,
                    index=vmsk,
                    src=vidf_outs
                )

            prop_scores_max_boxes = prop_scores_max_boxes * srl_arg_inds_msk

            # # B x num_cmp x num_srl_args
            # m1 = prop_scores_max_boxes == 0
            # prop_scores_max_boxes[m1] = 1
            # fin_scores = prop_scores_max_boxes.prod(
            #     dim=-1)
            # fin_scores = fin_scores * (m1.float().sum(dim=-1) > 0).float()

            # # b x num_cmp
            fin_scores_eval = prop_scores_max_boxes.sum(
                dim=-1) / srl_arg_inds_msk.sum(dim=-1)

            verb_msk = inp['num_cmp_msk']
            fin_scores_eval = fin_scores_eval * verb_msk.float()

            fin_scores_loss = prop_scores_max_boxes * verb_msk.unsqueeze(
                -1).expand(*prop_scores_max_boxes.shape).float()
            return {
                # B x num_cmp
                'fin_scores_eval': fin_scores_eval,
                # B x num_cmp x num_srl_args
                'fin_scores_loss': fin_scores_loss
            }

        else:
            # B x num_cmp x num_cmp x num_srl_args
            prop_scores_max_boxes, _ = torch.max(prop_scores, dim=-1)
            # B x num_cmp x num_cmp
            fin_scores = prop_scores_max_boxes.sum(dim=-1)
        return fin_scores

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

        # B x num_cmp x 5 x 512
        srl_arg_lstm_encoded = self.retrieve_srl_arg_from_lang_encode(
            lstm_encoded, inp
        )

        # Get visual features
        # B x num_cmp x 1000 x 512
        prop_feats = self.prop_feats_encode(inp)
        B, num_cmp, num_props, pdim = prop_feats.shape

        # Get seg features
        # B x num_cmp x 10 x 512
        seg_feats = self.seg_feats_encode(inp)

        # B x num_cmp x nfrm*nppf x psdim
        prop_seg_feats = self.concat_prop_seg_feats(prop_feats, seg_feats, inp)

        prop_seg_feats = self.simple_obj_interact_input(
            prop_seg_feats, inp
        )
        # Concat lang + region feats
        # conc_feats = self.concate_vis_lang_feats(
        #     prop_feats, srl_arg_lstm_encoded)
        # B x num_cmp x num_srl_args x num_propsx vf+lf dim

        last_hidden = lstm_outs['final_hidden'].view(
            B, num_verbs, lstm_outs['final_hidden'].size(-1)
        )

        verb_feats = self.retrieve_verb_lang_encoding(
            srl_arg_lstm_encoded, inp
        )

        srl_arg_lstm_encoded = self.simple_srl_attn(
            srl_arg_lstm_encoded, last_hidden, verb_feats, inp
        )

        if srl_arg_lstm_encoded.size(1) == 1 and num_cmp > 1:
            srl_arg_lstm_encoded = srl_arg_lstm_encoded.expand(
                -1, num_cmp, -1, -1
            )

        conc_feats = self.concate_vis_lang_feats(
            prop_seg_feats, srl_arg_lstm_encoded
        )

        # B x num_cmp x num_srl_args x num_props
        conc_feats_out_dict = self.conc_encode(conc_feats, inp)
        conc_feats_out = conc_feats_out_dict['conc_feats_out']

        seg_feats_for_verb, verb_feats = self.get_seg_verb_feats_to_process(
            seg_feats, srl_arg_lstm_encoded, lstm_outs, inp
        )

        if verb_feats.size(1) == 1 and num_cmp > 1:
            verb_feats = verb_feats.expand(-1, num_cmp, -1)

        # B x num_cmp
        vidf_outs = self.compute_seg_verb_feats_out(
            seg_feats_for_verb, verb_feats
        )
        fin_scores = self.compute_fin_scores(
            conc_feats_out_dict, inp, vidf_outs
        )

        num_cmp_msk = inp['num_cmp_msk'].view(
            B, num_cmp, 1, 1
        ).expand(
            B, num_cmp, num_srl_args,
            self.num_sampled_frm * self.num_prop_per_frm
        ).contiguous(
        ).view(*conc_feats_out.shape)

        srl_ind_msk = inp['srl_arg_inds_msk']
        if srl_ind_msk.size(1) == 1 and num_cmp > 1:
            srl_ind_msk = srl_ind_msk.expand(
                -1, num_cmp, -1, -1
            )
        srl_ind_msk = srl_ind_msk.unsqueeze(-1).expand(
            *conc_feats_out.shape)
        mdl_outs_eval = torch.sigmoid(
            conc_feats_out) * srl_ind_msk.float() * num_cmp_msk.float()

        return {
            'mdl_outs': conc_feats_out,
            'mdl_outs_eval': mdl_outs_eval,
            'vidf_outs': vidf_outs,
            'fin_scores_loss': fin_scores['fin_scores_loss'],
            'fin_scores': fin_scores['fin_scores_eval']
        }


class LossB_DS4(nn.Module):
    """
    Current DS4 model naive:
    Basically, if not correct, give score of 0
    """

    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        # self.loss_keys = ['loss', 'mdl_out_loss', 'vidf_loss']
        self.loss_keys = ['loss', 'mdl_out_loss', 'vidf_loss', 'verb_loss']
        self.loss_lambda = self.cfg.loss.loss_lambda
        # self.loss_fn = nn.BCEWithlogitsloss()
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
        # targets = overlaps.argmax(dim=-1, keepdim=True) * to_consider.long()

        srl_boxes = inp['srl_boxes']
        B, num_verbs, num_srl_args, num_box_per_srl = srl_boxes.shape
        B, num_cmp, num_props, num_gt_box = targets.shape

        if num_verbs == 1 and num_cmp > 1:
            srl_boxes = srl_boxes.expand(-1, num_cmp, -1, -1)

        srl_boxes_reshaped = srl_boxes.view(
            B, num_cmp, num_srl_args, 1, num_box_per_srl).expand(
                B, num_cmp, num_srl_args, num_props, num_box_per_srl)

        targets_reshaped = targets.view(
            B, num_cmp, 1, num_props, num_gt_box).expand(
                B, num_cmp, num_srl_args, num_props, num_gt_box)

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

        assert len(pnt_msk.shape) == 3
        # try:
        B = pad_props.size(0)
        num_cmp = pad_props.size(1)
        pad_props = combine_first_ax(pad_props)
        gt_bboxs = combine_first_ax(gt_bboxs)
        frm_msk = combine_first_ax(frm_msk)

        pnt_msk = combine_first_ax(pnt_msk)

        overlaps = bbox_overlaps(
            pad_props, gt_bboxs,
            (frm_msk | pnt_msk[:, :].unsqueeze(-1)))
        overlaps = overlaps.view(B, num_cmp, *overlaps.shape[1:])
        # except:
        #     import pdb
        #     pdb.set_trace()
        #     overlaps = bbox_overlaps(
        #         pad_props, gt_bboxs,
        #         (frm_msk | pnt_msk[:, 1:].unsqueeze(-1)))

        return overlaps

    def compute_loss_targets(self, inp):
        """
        Compute the targets, based on iou
        overlaps
        """
        overlaps = self.compute_overlaps(inp)
        B, ncmp, nprop, ngt = overlaps.shape
        overlaps_msk = overlaps.new_zeros(*overlaps.shape)

        # targ_cmp = inp['target_cmp'][0].item()
        targ_cmp = inp['target_cmp']
        # overlaps_msk[:, targ_cmp, ...] = 1
        overlaps_msk.scatter_(
            dim=1,
            index=targ_cmp.view(B, 1, 1, 1).expand(B, ncmp, nprop, ngt),
            src=overlaps_msk.new_ones(*overlaps_msk.shape)
        )

        overlaps_one_targ = overlaps * overlaps_msk

        targets_one = self.get_targets_from_overlaps(overlaps_one_targ, inp)
        targets_all = self.get_targets_from_overlaps(overlaps, inp)
        return {
            'targets_one': targets_one,
            'targets_all': targets_all
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

        # B x num_cmp
        num_cmp_msk = inp['num_cmp_msk']
        num_cmp = num_cmp_msk.size(1)
        srl_arg_boxes_mask = inp['srl_arg_boxes_mask']
        num_verbs = srl_arg_boxes_mask.size(1)
        if num_verbs == 1 and num_cmp > 1:
            srl_arg_boxes_mask = srl_arg_boxes_mask.expand(-1, num_cmp, -1)

        B, num_cmp, num_srl_args = srl_arg_boxes_mask.shape

        # boxes_msk = srl_arg_boxes_mask.float() * num_cmp_msk.unsqueeze(
        # -1).expand(*srl_arg_boxes_mask.shape).float()
        boxes_msk = num_cmp_msk.unsqueeze(
            -1).expand(*srl_arg_boxes_mask.shape).float()

        # B x num_cmp x num_srl_args -> B x num_cmp x num_srl x 1000
        boxes_msk = boxes_msk.unsqueeze(
            -1).expand(*targets_one.shape)

        tot_loss = tot_loss * boxes_msk

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
        mult = 0.1
        # B x ncmp
        msk = inp['num_cmp_msk']  #
        out_loss = torch.masked_select(out_loss.sum(dim=-1) * msk.float(),
                                       msk.byte()) * mult
        return out_loss.mean()

    def compute_vidf_loss(self, vidf_outs, inp):
        B, num_cmp, num_srl_args = vidf_outs.shape
        box_msk = inp['srl_arg_boxes_mask']
        srl_arg_ind_msk = inp['srl_arg_inds_msk']
        vidf_outs = ((vidf_outs * box_msk.float()).sum(dim=-1) /
                     srl_arg_ind_msk.sum(dim=-1).float())
        vidf_targs = vidf_outs.new_zeros(*vidf_outs.shape)

        targ_cmp = inp['target_cmp']

        vidf_targs.scatter_(
            dim=1,
            index=targ_cmp.unsqueeze(-1).expand(*vidf_targs.shape),
            src=vidf_targs.new_ones(*vidf_targs.shape)
        )

        # targ_cmp = inp['target_cmp'][0].item()
        # vidf_targs[:, targ_cmp] = 1
        vidf_loss = F.binary_cross_entropy(  #
            vidf_outs, vidf_targs,
            reduction='none'
        )
        msk = inp['num_cmp_msk']
        vidf_loss = vidf_loss * msk.float()
        vidf_loss = torch.masked_select(vidf_loss, msk.byte())
        return vidf_loss.mean()
        # vidf_loss = F.binary_cross_entropy_with_logits(
        #     vidf_outs,
        #     # inp['verb_cross_cmp'].float(),
        #     vidf_targets,
        #     reduction='none') * vidf_outs.size(1) * vidf_outs.size(2)

        # vidf_loss = torch.masked_select(
        #     vidf_loss, inp['verb_cross_cmp_msk'].byte())
        # vidf_loss = vidf_loss.mean()
        # return vidf_loss

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

        # # B x num_cmp x num_cmp x 1
        # vidf_outs = out['vidf_outs']
        # vidf_targets = inp['verb_cross_cmp'].float()
        # vidf_outs = out['fin_scores']

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

        # vidf_loss = mdl_out_loss.new_zeros(mdl_out_loss.shape)
        vidf_outs = out['fin_scores_loss']
        vidf_loss = self.compute_vidf_loss(vidf_outs, inp)
        out_loss = mdl_out_loss + verb_loss
        # if not self.cfg.mdl.loss.only_vid_loss:

        # out_loss = mdl_out_loss + verb_loss
        # else:
        # out_loss = vidf_loss
        out_loss_dict = {
            'loss': out_loss,
            'mdl_out_loss': mdl_out_loss,
            'vidf_loss': vidf_loss,
            'verb_loss': verb_loss
        }

        return {k: v * self.loss_lambda for k, v in out_loss_dict.items()}
