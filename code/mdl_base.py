"""
Base Model and Loss
Other models build on top of this.
Basically, have all the required args here.
"""
from torch import nn
from munch import Munch


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
        self.build_model()

    def build_model(self):
        self.build_lang_model()
        self.build_vis_model()
        self.build_conc_model()

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

    def build_lang_model(self):
        """
        How to encode the input sentence
        """
        raise NotImplementedError

    def build_vis_model(self):
        """
        How to encode the visual features
        How to encode proposal features
        and rgb, motion features
        """
        raise NotImplementedError

    def build_conc_model(self):
        """
        How to concatenate language and visual features
        """
        raise NotImplementedError


def main():
    from _init_stuff import Fpath, Arr, yaml
    from yacs.config import CfgNode as CN
    from dat_loader_simple import get_data
    cfg = CN(yaml.safe_load(open('./configs/anet_srl_cfg.yml')))
    data = get_data(cfg)
    comm = data.train_dl.dataset.comm
    mdl = AnetBaseMdl(cfg, comm)
    return mdl


if __name__ == '__main__':
    main()
