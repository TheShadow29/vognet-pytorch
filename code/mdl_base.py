"""
Base Model and Loss
Other models build on top of this.
"""
import torch
from torch import nn
from munch import Munch
from mdl_srl_utils import LSTMEncoder


class AnetBaseMdl(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        # Common stuff that needs to be passed around
        if comm is not None:
            assert isinstance(comm, (dict, Munch))
            self.comm = Munch(comm)
        else:
            self.comm = Munch()

        self.set_args()
        self.after_init()

    def after_init(self):
        pass

    def set_args(self):
        """
        Place to set all the required arguments, taken from cfg
        """
        # Vocab size needs to be in the ds
        # Can be added after after creation of the DATASET
        self.vocab_size = self.comm.vocab_size

        # Number of object classes
        # Also after creation of dataset.
        # Perhaps a good idea to keep all stuff
        # to be passed from ds to mdl in a separate
        # argument. Could be really helpful
        self.detect_size = self.comm.detect_size

        # Input encoding size
        # This is the size of the embedding for each word
        self.input_encoding_size = self.cfg.mdl.input_encoding_size

        # Hidden dimension of RNN
        self.rnn_size = self.cfg.mdl.rnn.rnn_size

        # Number of layers in RNN
        self.num_layers = self.cfg.mdl.rnn.num_layers

        # Dropout probability of LM
        self.drop_prob_lm = self.cfg.mdl.rnn.drop_prob_lm

        # itod
        self.itod = self.comm.itod

        self.num_sampled_frm = self.cfg.ds.num_sampled_frm
        self.num_prop_per_frm = self.comm.num_prop_per_frm

        self.unk_idx = int(self.comm.wtoi['UNK'])

        # Temporal attention size
        self.t_attn_size = self.cfg.ds.t_attn_size

        # srl_arg_len
        self.srl_arg_len = self.cfg.misc.srl_arg_length


class AnetSimpleBCEMdlDS4(AnetBaseMdl):
    def set_args(self):
        AnetBaseMdl.set_args(self)
        self.set_args_mdl()
        self.set_args_conc()

    def set_args_mdl(self):
        """
        Mdl specific args
        """
        return

    def set_args_conc(self):
        """
        Conc Type specific args
        """
        return

    def after_init(self):
        self.build_model()

    def build_model(self):
        self.build_lang_model()
        self.build_vis_model()
        self.build_conc_model()

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
        # B*6 x 40
        return {
            'src_tokens': srl_out_arg_seq,
            'src_tags': srl_tag
        }

    def retrieve_srl_arg_from_lang_encode(self, lstm_encoded, inp):
        """
        lstm_encoding: B*6 x 40 x 2048
        output: B*6 x 5 x 4096
        Basically, given the lstm inputs, want to separate out just
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
            out_srl_words_encoded)

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
        How to encode the visual features
        How to encode proposal features
        and rgb, motion features
        """
        pass

    def build_conc_model(self):
        """
        How to concatenate language and visual features
        """
        pass

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

    def get_seg_verb_feats_to_process(
            self,
            seg_feats, srl_arg_lstm_encoded,
            lstm_outs, inp):
        """
        Convenience function to make lesser
        clusterfuck.
        AI is IF-ELSE statements :)
        """
        B, num_verbs, num_srl_args, seq_len = inp['srl_arg_words_ind'].shape
        # num_cmp = seg_feats.size(1)
        seg_feats_for_verb = seg_feats.mean(dim=-2)

        # Use full sentence features
        verb_feats = lstm_outs['final_hidden']  #
        B_num_cmp, ldim = verb_feats.shape
        verb_feats = verb_feats.view(B, num_verbs, ldim)
        return seg_feats_for_verb, verb_feats

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

    def conc_encode(self, conc_feats, inp):
        """
        How to output from the concatenated features
        """
        pass

    def forward(self, inp):
        """
        Forward pass
        """
        pass


def main():
    from _init_stuff import Fpath, Arr, yaml
    from yacs.config import CfgNode as CN
    from dat_loader_simple import get_data
    cfg = CN(yaml.safe_load(open('./configs/anet_srl_cfg.yml')))
    data = get_data(cfg)
    comm = data.train_dl.dataset.comm
    mdl = AnetSimpleBCEMdlDS4(cfg, comm)
    return mdl


if __name__ == '__main__':
    main()
