"""
Different Models: ImgGrnd, VidGrnd, VOGNet
"""
import torch
from torch import nn
from mdl_base import AnetSimpleBCEMdlDS4
from mdl_conc import ConcSEP, LossB_DS4
from mdl_conc_single import (
    ConcTemp, LossB_SSJ1_Temporal_DS4,
    ConcSPAT, LossB_SSJ1_Spatial_DS4
)
from mdl_srl_utils import do_cross
from transformer_code import Transformer, RelTransformer


class ImgGrnd(AnetSimpleBCEMdlDS4):
    def set_args_mdl(self):
        # AnetSimpleBCEMdlDS4.set_args(self)
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

    def build_vis_model(self):
        self.prop_encoder = nn.Sequential(
            *[nn.Linear(self.prop_dim, self.prop_encode_dim),
              nn.ReLU()])
        self.seg_encoder = nn.Sequential(
            *[nn.Linear(self.seg_feat_dim, self.seg_feat_encode_dim),
              nn.ReLU()])

        self.seg_verb_classf = nn.Sequential(
            *[
                nn.Linear(self.seg_feat_encode_dim+self.lang_encode_dim,
                          256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ]
        )

    def build_conc_model(self):
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

    def simple_obj_interact(self, ps_feats, inp, ncmp, nfrm, nppf):
        """
        For ImgGrnd, no object interacation
        """
        return ps_feats


class ImgGrnd_SEP(ConcSEP, ImgGrnd):
    pass


class ImgGrnd_TEMP(ConcTemp, ImgGrnd):
    pass


class ImgGrnd_SPAT(ConcSPAT, ImgGrnd):
    pass


class VidGrnd(ImgGrnd):
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


class VidGrnd_TEMP(ConcTemp, VidGrnd):
    pass


class VidGrnd_SPAT(ConcSPAT, VidGrnd):
    pass


class VOGNet(VidGrnd):
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

        self.pe_mul_sub_enc = nn.Sequential(
            *[
                nn.Linear(5, n_heads),
                nn.ReLU(),
            ]
        )

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
            # conc_feats_out = self.lin2(conc_feats).squeeze(-1)
        else:
            conc_feats = conc_feats_sa.view(
                B, ncmp, nsrl, nprop, vldim
            ).contiguous()
            # conc_feats_out = self.lin2(conc_feats).squeeze(-1)
        return {
            'conc_feats_out': conc_feats,
        }


class VOG_SEP(ConcSEP, VOGNet):
    pass


class VOG_TEMP(ConcTemp, VOGNet):
    pass


class VOG_SPAT(ConcSPAT, VOGNet):
    pass
