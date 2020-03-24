"""
Different Models: ImgGrnd, VidGrnd, VOGNet

Also contains shape comments for each
conc strategy, also see forward functions
for SEP and TEMP for completenes

Note that Language Encoding is the same for all

Assume:
- 4 videos, one of them groundtruth
- each video with 10 frames
- each frame with 5 proposals
"""

import torch
from torch import nn
from mdl_base import AnetBaseMdl
from mdl_conc_sep import ConcSEP, LossB_SEP
from mdl_conc_single import (
    ConcTEMP, LossB_TEMP,
    ConcSPAT, LossB_SPAT
)
from mdl_srl_utils import do_cross
from transformer_code import Transformer, RelTransformer
from mdl_srl_utils import LSTMEncoder


class ImgGrnd(AnetBaseMdl):
    """
    ImgGrnd model. Implements basic language stuff
    and directly uses prop+seg feats with language

    VidGrnd and VOGNet improve over this with Object Tx
    and MultiModal Tx with Relative Position Encoding

    Forward function is implemented in ConcSEP and ConcTEMP
    since they are slightly different use cases
    (multiple videos vs single videos)
    """

    def set_args_mdl(self):
        """
        Model specific args
        """
        # proposal dimension
        self.prop_dim = self.cfg.mdl.prop_feat_dim
        # Encoded dimension of the region features
        self.prop_encode_dim = self.cfg.mdl.vsrl.prop_encode_size

        # Segment features (2048+1024)
        self.seg_feat_dim = self.cfg.mdl.seg_feat_dim
        # Encoded dimension of the segment features
        self.seg_feat_encode_dim = self.cfg.mdl.vsrl.seg_encode_size

        # Feature projection of the captured language vectors
        self.lang_encode_dim = self.cfg.mdl.vsrl.lang_encode_size

        self.prop_seg_feat_dim = (
            self.prop_encode_dim + self.seg_feat_encode_dim
        )

        self.vis_lang_feat_dim = self.prop_seg_feat_dim + self.lang_encode_dim

        self.conc_encode_item = getattr(self, 'conc_encode_simple')

    def get_srl_arg_seq_to_sent_seq(self, inp):
        """
        srl_arg_seq: B x 6 x 5 x 40
        output: B x 6 x 40
        Input is like [ARG0-> wlist1, V->wlist2...]
        Output is like [wlist1..wlist2...]
        """
        srl_arg_seq = inp['srl_arg_words_ind']
        B, num_verbs, num_srl_args, seq_len = srl_arg_seq.shape
        srl_arg_seq_reshaped = srl_arg_seq.view(
            B*num_verbs, num_srl_args*seq_len
        )

        srl_arg_word_mask = inp['srl_arg_word_mask'].view(B*num_verbs, -1)
        msk = srl_arg_word_mask == -1
        srl_arg_word_mask[msk] = 0
        srl_out_arg_seq = torch.gather(
            srl_arg_seq_reshaped, dim=1, index=srl_arg_word_mask
        )

        srl_out_arg_seq[msk] = self.vocab_size

        srl_tag = inp['srl_tag_word_ind'].view(B*num_verbs, -1)
        assert srl_tag.shape == srl_out_arg_seq.shape

        return {
            'src_tokens': srl_out_arg_seq,
            'src_tags': srl_tag
        }

    def retrieve_srl_arg_from_lang_encode(self, lstm_encoded, inp):
        """
        lstm_encoding: B*6 x 40 x 2048
        output: B*6 x 5 x 4096

        Basically, given the lstm inputs,
        want to separate out just
        the argument parts
        """
        def gather_from_index(inp1, dim1, index1):
            index1_reshaped = index1.unsqueeze(
                -1).expand(*index1.shape, inp1.size(-1))
            return torch.gather(inp1, dim1, index1_reshaped)

        # B x 6 x 5 x 2
        srl_arg_words_capture = inp['srl_arg_words_capture']
        B, num_verbs, num_srl_args, st_end = srl_arg_words_capture.shape
        assert st_end == 2
        srl_arg_words_capture = srl_arg_words_capture.view(
            B*num_verbs, num_srl_args, st_end
        )

        # B*num_verbs x 5 x 2048
        st_srl_words = gather_from_index(
            lstm_encoded, 1, srl_arg_words_capture[..., 0])
        end_srl_words = gather_from_index(
            lstm_encoded, 1, srl_arg_words_capture[..., 1])

        # concat start, end
        # B*num_verbs x 5 x 4096
        srl_words_encoded = torch.cat([st_srl_words, end_srl_words], dim=2)
        out_srl_words_encoded = srl_words_encoded.view(
            B, num_verbs, num_srl_args, -1)

        out_srl_words_encoded = self.srl_arg_words_out_enc(
            out_srl_words_encoded
        )

        # zero out which are not arg words
        # B x num_cmp x num_srl_args
        srl_arg_msk = inp['srl_arg_inds_msk']
        out_srl_words_encoded = out_srl_words_encoded * srl_arg_msk.unsqueeze(
            -1).expand(*out_srl_words_encoded.shape).float()
        return out_srl_words_encoded

    def retrieve_verb_lang_encoding(self, lang_encoding, inp):
        """
        lang_encoding: B x num_cmp x 5 (num_srl_args) x ldim (512)
        output: B x num_cmp x ldim
        Basically, choose the srl_argument which corresponds
        to the VERB
        """
        verb_inds = inp['verb_ind_in_srl']
        _, _, num_srl_args, ldim = lang_encoding.shape
        B, num_cmp = verb_inds.shape
        verb_lang_enc = torch.gather(
            lang_encoding,
            dim=-2,
            index=verb_inds.view(B, num_cmp, 1, 1).expand(
                B, num_cmp, 1, ldim)
        )
        return verb_lang_enc.squeeze(-2)

    def build_lang_model(self):
        """
        How to encode the input sentence
        """
        # LSTM process
        self.lstm_encoder = LSTMEncoder(
            cfg=self.cfg,
            comm=self.comm,
            embed_dim=self.input_encoding_size,
            hidden_size=self.rnn_size,
            num_layers=self.num_layers,
            bidirectional=True,
            left_pad=False,
            num_embeddings=self.vocab_size+1,
            pad_idx=self.vocab_size
        )

        # After passing through lstm, we collect
        # first and last word of the argument and concatenate
        # The following is a feature projection after that step
        # *2 because of bidirectional, *2 because of first/last
        # word concatenation
        self.lstm_out_feat_proj = nn.Sequential(
            *[nn.Linear(self.rnn_size*2, self.lang_encode_dim),
              nn.ReLU()])

        self.srl_arg_words_out_enc = nn.Sequential(
            *[nn.Linear(self.lang_encode_dim*2, self.lang_encode_dim),
              nn.ReLU()])

        self.srl_simple_lin = nn.Sequential(
            *[nn.Linear(self.lang_encode_dim * 3, self.lang_encode_dim),
              nn.ReLU()]
        )

    def build_vis_model(self):
        """
        Need to encode the proposal features,
        and segment features.
        Also, for sep we need additional seg+verb loss
        Not used for others
        """
        self.prop_encoder = nn.Sequential(
            *[nn.Linear(self.prop_dim, self.prop_encode_dim),
              nn.ReLU()])
        self.seg_encoder = nn.Sequential(
            *[nn.Linear(self.seg_feat_dim, self.seg_feat_encode_dim),
              nn.ReLU()])

        # Only used for SEP
        # Not for others
        self.seg_verb_classf = nn.Sequential(
            *[
                nn.Linear(self.seg_feat_encode_dim+self.lang_encode_dim,
                          256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ]
        )

    def build_conc_model(self):
        """
        how to encode Vis+Lang features
        """
        self.lin2 = nn.Sequential(
            *[
                nn.Linear(self.vis_lang_feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ]
        )
        self.lin_tmp = nn.Sequential(
            *[
                nn.Linear(self.vis_lang_feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ]
        )

    def simple_srl_attn(self, q0_srl, q0, q0_verb, inp):
        B, nv, nsrl, qdim = q0_srl.shape
        assert q0.size(-1) == qdim
        q0_srl_cat = torch.cat([
            q0_srl,
            q0.view(B, nv, 1, qdim).expand(B, nv, nsrl, qdim),
            q0_verb.view(B, nv, 1, qdim).expand(B, nv, nsrl, qdim),
        ], dim=-1)
        # B x nv x nsrl x 2*qdim
        return self.srl_simple_lin(q0_srl_cat)

    def lang_encode(self, src_tokens_tags, src_lens):
        """
        Encodes the input sentence
        """
        src_lens = src_lens.squeeze(-1)
        src_tokens = src_tokens_tags['src_tokens']
        # src_tags = src_tokens_tags['src_tags']
        src_tokens = src_tokens[:, :src_lens.max().item()].contiguous()
        # if self.cfg.mdl.lang_use_tags:
        # src_tags = src_tags[:, :src_lens.max().item()].contiguous()
        # else:
        # src_tags = None

        # the output is a dictioary of 'encoder_out',
        # 'encoder_padding_mask', the latter is not used
        # 'encoder_out' is (full output, final hidden, final cells)

        lstm_out = self.lstm_encoder(src_tokens, src_lens)

        lstm_full_out, final_hidden, final_cells = lstm_out['encoder_out']

        # B*num_cmp x seq_len x 2048
        lstm_full_output = self.lstm_encoder.reorder_only_outputs(
            lstm_full_out)

        lstm_full_output = self.lstm_out_feat_proj(lstm_full_output)

        # choose last layer outputs
        hidden_out = self.lstm_out_feat_proj(final_hidden[-1])

        return {
            'lstm_full_output': lstm_full_output,
            'final_hidden': hidden_out
        }

    def simple_obj_interact(self, ps_feats, inp, ncmp, nfrm, nppf):
        """
        For ImgGrnd, no object interacation
        """
        return ps_feats

    def prop_feats_encode(self, inp):
        """
        Encoding the proposal features.
        """
        # B x num_cmp x 1000 x 2048
        prop_feats = inp['pad_region_feature']
        # B x num_cmp x 1000 x 512
        prop_feats_out = self.prop_encoder(prop_feats)
        return prop_feats_out

    def seg_feats_encode(self, inp):
        """
        Encoding segment features
        """
        # # B x num_cmp x 480 x 3072
        # seg_feats = inp['seg_feature']

        # B x num_cmp x 10 x 3072
        seg_feats = inp['seg_feature_for_frms']
        # # B x num_cmp x 480 x 512

        # B x num_cmp x 10 x 512
        seg_feats_out = self.seg_encoder(seg_feats)
        return seg_feats_out

    def concate_vis_lang_feats(self, vis_feats, lang_feats, do='concat'):
        """
        Concatenate visual and language features
        vis_feats: B x num_cmp x 1000 x 2048 (last dim could be different)
        lang_feats: B x num_cmp x 5 x 4096
        output: concatenated features of shape B x num_cmp x 5 x 1000 x (2048+4096)
        """

        B, num_cmp_v, num_props, vf_dim = vis_feats.shape
        B, num_cmp_l, num_srl_args, lf_dim = lang_feats.shape
        assert num_cmp_v == num_cmp_l
        num_cmp = num_cmp_v
        # expand visual features
        out_feats_vis = vis_feats.view(
            B, num_cmp, 1, num_props, vf_dim).expand(
            B, num_cmp, num_srl_args, num_props, vf_dim)

        # expand language features
        out_feats_lang = lang_feats.view(
            B, num_cmp, num_srl_args, 1, lf_dim
        ).expand(
            B, num_cmp, num_srl_args, num_props, lf_dim
        )
        if do == 'concat':
            # B x num_cmp x num_srl_args x num_props x (vf_dim + lf_dim)
            return torch.cat([out_feats_vis, out_feats_lang], dim=-1)
        elif do == 'none':
            # B x num_cmp x num_srl_args x num_propsx vf/lf dim
            return out_feats_vis, out_feats_lang

    def conc_encode_simple(self, conc_feats, inp, nfrm, nppf, ncmp):
        """
        conc_feats: B x 6 x 5 x 1000 x 6144
        output: B x 6 x 5 x 1000 x 1
        """
        B, ncmp1, nsrl, nprop, vldim = conc_feats.shape
        assert ncmp1 == ncmp
        conc_feats_out = self.lin2(conc_feats)
        conc_feats_temp = conc_feats.view(
            B, ncmp, nsrl, nfrm,
            nppf, vldim
        ).sum(dim=-2)
        # B x ncmp x nsrl x nfrms x (vldim->1)
        conc_temp_out = self.lin_tmp(conc_feats_temp)
        return {
            'conc_feats_out': conc_feats_out.squeeze(-1),
            'conc_temp_out': conc_temp_out.squeeze(-1)
        }

    def get_seg_verb_feats_to_process(
            self,
            seg_feats, srl_arg_lstm_encoded,
            lstm_outs, inp):
        """
        Convenience function to make lesser
        clusterfuck.
        """
        B, num_verbs, num_srl_args, seq_len = inp['srl_arg_words_ind'].shape
        # num_cmp = seg_feats.size(1)
        seg_feats_for_verb = seg_feats.mean(dim=-2)

        # Use full sentence features
        verb_feats = lstm_outs['final_hidden']  #
        B_num_cmp, ldim = verb_feats.shape
        verb_feats = verb_feats.view(B, num_verbs, ldim)
        return seg_feats_for_verb, verb_feats

    def compute_seg_verb_feats_out(self, seg_feats, verb_feats):
        """
        seg_feats: B x num_cmp x 512
        verb_feats: B x num_cmp x 512
        """
        # B x num_cmp x 512

        B, num_cmp, ldim = verb_feats.shape
        seg_verb_feats = torch.cat([
            verb_feats, seg_feats
        ], dim=-1)

        seg_verb_feats_outs = self.seg_verb_classf(seg_verb_feats)
        # B x num_cmp
        return seg_verb_feats_outs.squeeze(-1)


class ImgGrnd_SEP(ConcSEP, ImgGrnd):
    pass


class ImgGrnd_TEMP(ConcTEMP, ImgGrnd):
    pass


class ImgGrnd_SPAT(ConcSPAT, ImgGrnd):
    pass


class VidGrnd(ImgGrnd):
    """
    Add Object Transformer to ImgGrnd
    """

    def build_vis_model(self):
        ImgGrnd.build_vis_model(self)
        n_layers = self.cfg.mdl.obj_tx.n_layers
        n_heads = self.cfg.mdl.obj_tx.n_heads
        attn_drop = self.cfg.mdl.obj_tx.attn_drop

        if self.cfg.mdl.obj_tx.use_rel:
            self.obj_txf = RelTransformer(
                self.prop_seg_feat_dim, 0, 0,
                d_hidden=int(self.prop_seg_feat_dim//2),
                n_layers=n_layers,
                n_heads=n_heads,
                drop_ratio=attn_drop,
                pe=False,
                d_pe=5
            )
        else:
            self.obj_txf = Transformer(
                self.prop_seg_feat_dim, 0, 0,
                d_hidden=int(self.prop_seg_feat_dim//2),
                n_layers=n_layers,
                n_heads=n_heads,
                drop_ratio=attn_drop,
                pe=False,
            )

        if self.cfg.mdl.obj_tx.use_ddp:
            self.obj_tx = nn.DataParallel(self.obj_txf)

        self.pe_obj_sub_enc = nn.Sequential(
            *[
                nn.Linear(5, n_heads),
                nn.ReLU(),
            ]
        )

        self.vid_w = self.cfg.ds.resized_width
        self.vid_h = self.cfg.ds.resized_height

    def compute_pe(self, props, nsrl, nfrm, nppf, ncmp,
                   with_cross=False, pe_enc=None):
        # B x ncmp x nprops x 7
        props[..., 0] /= self.vid_w
        props[..., 1] /= self.vid_h
        props[..., 2] /= self.vid_w
        props[..., 3] /= self.vid_h
        props[..., 4] /= nfrm
        if (self.cfg.ds.conc_type == 'spat' or
                self.cfg.ds.conc_type == 'temp'):
            B, nprops, pdim = props.shape
        else:
            B, ncmp1, nprops, pdim = props.shape
            assert ncmp1 == ncmp
        assert nfrm * nppf == nprops
        # if self.cfg.mdl.interact.per_frm or self.cfg.mdl.interact.cross_frm:
        props1 = props.view(
            B*ncmp * nfrm, nppf, pdim
        )
        # if with_cross:
        # B*ncmp*nfrm x nppf x nppf x pdim
        props_subt = do_cross(props1, dim1=-2, op='subtract')
        if pe_enc is None:
            pe_enc = self.pe_sub_enc
        props_subt = pe_enc(props_subt)
        n_heads = props_subt.size(-1)
        props_subt = props_subt.view(
            B*ncmp, nfrm, 1, nppf, 1, nppf, n_heads
        ).expand(
            B*ncmp, nfrm, nsrl, nppf, nsrl, nppf, n_heads
        ).contiguous().view(
            B*ncmp*nfrm, nsrl*nppf, nsrl*nppf, n_heads
        )

        return props_subt

    def simple_obj_interact(self, ps_feats, inp, ncmp, nfrm, nppf):
        B, num_cmp1, nprops, psdim = ps_feats.shape
        assert ncmp == num_cmp1
        assert nprops == nfrm * nppf
        if self.cfg.mdl.obj_tx.one_frm:
            props = inp['pad_proposals'][..., :5].clone().detach()
            pe_props = self.compute_pe(
                props, 1, nfrm, nppf, ncmp, with_cross=True,
                pe_enc=self.pe_obj_sub_enc
            )
            ps_feats_pre = ps_feats.view(
                B*ncmp*nfrm, nppf, psdim
            ).contiguous()
        else:
            props = inp['pad_proposals'][..., :5].clone().detach()
            pe_props = self.compute_pe(
                props, 1, 1, nprops, ncmp,
                with_cross=True, pe_enc=self.pe_obj_sub_enc
            )
            ps_feats_pre = ps_feats.view(
                B*ncmp, nprops, psdim
            ).contiguous()

        if self.cfg.mdl.obj_tx.use_rel:
            ps_feats_sa = self.obj_txf(ps_feats_pre, pe_props)
        else:
            ps_feats_sa = self.obj_txf(ps_feats_pre)
        prop_seg_feats = ps_feats_sa.view(
            B, ncmp, nprops, psdim
        )

        return prop_seg_feats


class VidGrnd_SEP(ConcSEP, VidGrnd):
    pass


class VidGrnd_TEMP(ConcTEMP, VidGrnd):
    pass


class VidGrnd_SPAT(ConcSPAT, VidGrnd):
    pass


class VOGNet(VidGrnd):
    """
    Add MultiModal Tx to VidGrnd
    """

    def set_args_mdl(self):
        VidGrnd.set_args_mdl(self)
        self.conc_encode_item = getattr(self, 'conc_encode_sa')

    def build_conc_model(self):
        VidGrnd.build_conc_model(self)
        n_layers = self.cfg.mdl.mul_tx.n_layers
        n_heads = self.cfg.mdl.mul_tx.n_heads
        attn_drop = self.cfg.mdl.mul_tx.attn_drop

        if self.cfg.mdl.mul_tx.use_rel:
            self.mult_txf = (
                RelTransformer(
                    self.vis_lang_feat_dim, 0, 0,
                    d_hidden=int(self.vis_lang_feat_dim//2),
                    n_layers=n_layers,
                    n_heads=n_heads,
                    drop_ratio=attn_drop,
                    pe=False,
                    d_pe=5
                )
            )
        else:
            self.mult_txf = (
                Transformer(
                    self.vis_lang_feat_dim, 0, 0,
                    d_hidden=int(self.vis_lang_feat_dim//2),
                    n_layers=n_layers,
                    n_heads=n_heads,
                    drop_ratio=attn_drop,
                    pe=False
                )
            )

        if self.cfg.mdl.mul_tx.use_ddp:
            self.mult_txf = nn.DataParallel(self.mult_txf)

        self.pe_mul_sub_enc = nn.Sequential(
            *[
                nn.Linear(5, n_heads),
                nn.ReLU(),
            ]
        )

    def simple_obj_interact(self, ps_feats, inp, ncmp, nfrm, nppf):
        if self.cfg.mdl.obj_tx.to_use:
            return VidGrnd.simple_obj_interact(
                self, ps_feats, inp, ncmp, nfrm, nppf
            )
        else:
            return ps_feats

    def conc_encode_sa(self, conc_feats, inp, nfrm, nppf, ncmp):
        def unpack(inp_t):
            """
            unpacks on dim=1
            """
            # inp_t = inp_t.view(B*ncmp*nsrl, nfrm-1, 2,
            # nppf, vldim).contiguous()
            out_t1 = torch.cat(
                [inp_t[:, :, 0, ...], inp_t[:, [-1], 1, ...]],
                dim=1
            )
            out_t2 = torch.cat(
                [inp_t[:, [0], 0, ...], inp_t[:, :, 1, ...]],
                dim=1
            )
            out_t = (out_t1 + out_t2) / 2
            return out_t

        def unpack_t(inp_t):
            out_t = torch.cat(
                [inp_t[:, :, 0, ...], inp_t[:, [-1], 0, ...]],
                dim=1
            )
            return out_t

            # return out_t.view(B, ncmp, nsrl, nfrm*nppf, vldim)

        assert self.cfg.mdl.mul_tx.one_frm or self.cfg.mdl.mul_tx.cross_frm

        pe_props = inp['pad_proposals'][..., :5].clone().detach()
        if self.cfg.mdl.mul_tx.one_frm:
            out_dict_pfrm = self.conc_encode2(
                conc_feats, inp, nfrm, nppf, ncmp, pe_props,
                int_pfrm=True
            )
        else:
            out_dict_pfrm = {'conc_feats_out': 0, 'conc_temp_out': 0}

        if self.cfg.mdl.mul_tx.cross_frm:
            B, ncmp1, nsrl, nprop, vldim = conc_feats.shape
            assert ncmp1 == ncmp
            conc_feats_1 = conc_feats.view(
                B, ncmp, nsrl, nfrm, nppf, vldim
            )

            c1 = torch.cat([conc_feats_1[:, :, :, :-1],
                            conc_feats_1[:, :, :, 1:]], dim=4)
            c1 = c1.view(B, ncmp, nsrl, (nfrm-1)*2*nppf, vldim)

            pe_cross = pe_props.view(B, ncmp, nfrm, nppf, 5)
            p1 = torch.cat([pe_cross[:, :, :-1], pe_cross[:, :, 1:]], dim=3)
            if ncmp == 1:
                p1 = p1.view(B, (nfrm-1) * 2*nppf, 5)
            else:
                p1 = p1.view(B, ncmp, (nfrm-1) * 2*nppf, 5)

            out_dict1 = self.conc_encode2(
                c1, inp, nfrm-1, 2*nppf, ncmp, p1, int_pfrm=True
            )
            c2 = unpack(
                out_dict1['conc_feats_out'].view(
                    B*ncmp*nsrl, nfrm-1, 2, nppf, vldim)
            ).view(B, ncmp, nsrl, nfrm*nppf, vldim)

            t2 = unpack_t(
                out_dict1['conc_temp_out'].view(
                    B*ncmp*nsrl, nfrm-1, 1, 1, vldim
                )
            ).view(B, ncmp, nsrl, nfrm, vldim)
            out_dict_cfrm = {
                'conc_feats_out': c2,
                # 'conc_temp_out': t2
            }
        else:
            out_dict_cfrm = {
                'conc_feats_out': 0,
                # 'conc_temp_out': 0
            }

        out_dict = {}
        out_dict['conc_feats_out'] = self.lin2(
            out_dict_pfrm['conc_feats_out'] + out_dict_cfrm['conc_feats_out']
        ).squeeze(-1)

        return out_dict

    def conc_encode2(self, conc_feats, inp, nfrm, nppf,
                     ncmp, pe_props, int_pfrm):
        """
        conc_feats: B x 6 x 5 x 1000 x 6144
        output: B x 6 x 5 x 1000 x 1
        """

        B, ncmp1, nsrl, nprop, vldim = conc_feats.shape
        assert ncmp1 == ncmp
        # if self.cfg.mdl.interact.per_frm:
        # return {'conc_feats_out': self.lin2(conc_feats).squeeze(-1)}
        if int_pfrm:
            conc_feats_sa_pre = conc_feats.view(
                B * ncmp, nsrl, nfrm,
                nppf, vldim
            ).transpose(1, 2).contiguous().view(
                B * ncmp * nfrm,
                nsrl * nppf, vldim
            ).contiguous()

        else:
            conc_feats_sa_pre = conc_feats.view(
                B * ncmp, nsrl * nprop, vldim
            ).contiguous()

        # B*ncmp x nfrm x nsrl*nppf x 5
        # pe = self.pe_enc(self.compute_pe(
            # inp, nsrl, nfrm, nppf, with_cross=False))
        #
        pe = self.compute_pe(
            pe_props, nsrl, nfrm, nppf, ncmp, with_cross=True,
            pe_enc=self.pe_mul_sub_enc
        )

        # Perform self-attn
        # B*ncmp x nfrm x nsrl*nppf x vldim
        # conc_feats_sa_pre += pe
        if self.cfg.mdl.mul_tx.use_rel:
            conc_feats_sa = self.mult_txf(conc_feats_sa_pre, pe)
        else:
            conc_feats_sa = self.mult_txf(conc_feats_sa_pre)
        # conc_feats_sa = self.obj_lang_interact(conc_feats_sa_pre)

        if int_pfrm:

            conc_feats_sa = conc_feats_sa.view(
                B * ncmp, nfrm, nsrl,
                nppf, vldim
            ).contiguous().transpose(
                1, 2).contiguous().view(
                    B*ncmp, nsrl, nfrm,
                    nppf, vldim
            ).contiguous()

            conc_feats = conc_feats_sa.view(
                B, ncmp, nsrl, nprop, vldim
            ).contiguous()
        else:
            conc_feats = conc_feats_sa.view(
                B, ncmp, nsrl, nprop, vldim
            ).contiguous()
        return {
            'conc_feats_out': conc_feats,
        }


class VOG_SEP(ConcSEP, VOGNet):
    pass


class VOG_TEMP(ConcTEMP, VOGNet):
    pass


class VOG_SPAT(ConcSPAT, VOGNet):
    pass
