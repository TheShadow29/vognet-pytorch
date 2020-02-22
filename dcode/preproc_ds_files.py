"""
Preprocess dataset files
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from yacs.config import CfgNode as CN
import numpy as np
import ast
from collections import Counter
# from torchtext import vocab
import pickle
from munch import Munch

np.random.seed(5)


class AnetCSV:
    def __init__(self, cfg, comm=None):
        self.cfg = cfg
        if comm is not None:
            assert isinstance(comm, (dict, Munch))
            self.comm = Munch(comm)
        else:
            self.comm = Munch()

        inp_anet_dict_fpath = cfg.ds.anet_ent_split_file
        self.inp_dict_file = Path(inp_anet_dict_fpath)

        # Create directory to keep the generated csvs
        self.out_csv_dir = self.inp_dict_file.parent / 'csv_dir'
        self.out_csv_dir.mkdir(exist_ok=True)

        # Structure of anet_dict:
        # anet = Dict,
        # keys: 1. word to lemma, 2. index to word,
        # 3. word to detection 4. video information
        # We only need the video information
        self.vid_dict_list = json.load(open(inp_anet_dict_fpath))['videos']

    def create_csvs(self):
        """
        Create the Train/Val split videos
        """
        self.vid_info_df = pd.DataFrame(self.vid_dict_list)
        self.vid_info_df.index.name = 'Index'

        train_df = self.vid_info_df[self.vid_info_df.split == 'training']
        train_df.to_csv(self.out_csv_dir / 'train.csv',
                        index=True, header=True)

        # NOTE: Test files don't have the annotations, so cannot be used.
        # Instead we split the validation dataframe into two parts (50/50)

        val_test_df = self.vid_info_df[self.vid_info_df.split == 'validation']

        # Randomly take half as validation, rest as test
        # Both are saved in val.csv, however, during evaluation
        # only those with "val" in the field "vt_split" are chosen
        msk = np.random.rand(len(val_test_df)) < 0.5
        val_test_df['vt_split'] = ['val' if m == 1 else 'test' for m in msk]
        val_test_df.to_csv(self.out_csv_dir / 'val.csv',
                           index=True, header=True)

    def post_proc(self, csv_file_type):
        """
        Some videos don't have features. These are removed
        for convenience.
        (only 4-5 videos were removed)
        """
        self.seg_feature_root = Path(self.cfg.ds.seg_feature_root)
        assert self.seg_feature_root.exists()

        self.feature_root = Path(self.cfg.ds.feature_root)
        assert self.feature_root.exists()

        csv_file = self.out_csv_dir / f'{csv_file_type}.csv'
        csv_df = pd.read_csv(csv_file)
        msk = []
        num_segs_list = []
        for row_ind, row in tqdm(csv_df.iterrows(), total=len(csv_df)):
            vid_seg_id = row['id']
            vid_id = row['vid_id']
            num_segs = csv_df[csv_df.vid_id == vid_id].seg_id.max() + 1
            num_segs_list.append(num_segs)

            vid_id_ix, seg_id_ix = vid_seg_id.split('_segment_')
            seg_rgb_file = self.seg_feature_root / \
                f'{vid_id_ix[2:]}_resnet.npy'
            seg_motion_file = self.seg_feature_root / f'{vid_id_ix[2:]}_bn.npy'
            region_feature_file = self.feature_root / f'{vid_seg_id}.npy'
            out = (seg_rgb_file.exists() and seg_motion_file.exists()
                   and region_feature_file.exists())
            msk.append(out)

        csv_df['num_segs'] = num_segs_list
        csv_df = csv_df[msk]
        csv_df.to_csv(self.out_csv_dir /
                      f'{csv_file_type}_postproc.csv', index=False, header=True)

    def post_proc_srl(self, train_file, val_file, test_file=None):
        """
        Add the Index to each csv file
        This helps later in contrastive sampling.
        Honestly, I forgot why I had to do it in such a roundabout way.
        Surely, there was a reason. May change later if a shortcut is found
        """
        def get_row_id(vid_seg, ann_df):
            vid_dict_row = ann_df[ann_df.id ==
                                  vid_seg]
            if len(vid_dict_row) == 1:
                vid_dict_row_id = vid_dict_row.index[0]
                return vid_dict_row_id
            else:
                return -1

        self.vid_info_df = pd.DataFrame(self.vid_dict_list)
        self.vid_info_df.index.name = 'Index'

        trn_ann_df = pd.read_csv(
            self.out_csv_dir / f'{train_file}_postproc.csv')
        val_ann_df = pd.read_csv(self.out_csv_dir / f'{val_file}_postproc.csv')

        srl_trn_val = pd.read_csv(self.cfg.ds.verb_ent_file)

        trn_ann_ind = []
        trn_msk = []

        val_ann_ind = []
        val_msk = []
        vt_msk = []

        for srl_ind, srl in tqdm(srl_trn_val.iterrows(),
                                 total=len(srl_trn_val)):
            req_args = ast.literal_eval(srl.req_args)
            if len(req_args) == 1:
                continue
            vid_seg = srl.vid_seg
            vid_dict_row = self.vid_info_df[self.vid_info_df.id == vid_seg]
            assert len(vid_dict_row) == 1
            vid_dict_row = vid_dict_row.iloc[0]
            split = vid_dict_row.split

            if split == 'training':
                ann_ind = get_row_id(vid_seg, trn_ann_df)
                if ann_ind == -1:
                    print(split, vid_seg)
                    continue
                trn_ann_ind.append(ann_ind)
                trn_msk.append(srl_ind)
            elif split == 'validation':
                ann_ind = get_row_id(vid_seg, val_ann_df)
                if ann_ind == -1:
                    print(split, vid_seg)
                    continue
                val_ann_ind.append(ann_ind)
                val_msk.append(srl_ind)
                vt_msk.append(val_ann_df.loc[ann_ind].vt_split)
            elif split == 'testing':
                pass
            else:
                raise NotImplementedError

        srl_trn = srl_trn_val.iloc[trn_msk]
        srl_trn['ann_ind'] = trn_ann_ind
        srl_trn['srl_ind'] = trn_msk
        srl_trn['vt_split'] = 'train'

        srl_val = srl_trn_val.iloc[val_msk]
        srl_val['ann_ind'] = val_ann_ind
        srl_val['srl_ind'] = val_msk
        srl_val['vt_split'] = vt_msk

        srl_trn.to_csv(self.cfg.ds.trn_verb_ent_file,
                       index=False, header=True)
        srl_val.to_csv(self.cfg.ds.val_verb_ent_file,
                       index=False, header=True)

    def process_arg_vocabs(self):
        def create_vocab(srl_annots, key):
            x_counter = Counter()
            for x_c in srl_annots[key]:
                x_counter += Counter(x_c)
            return vocab.Vocab(x_counter, specials_first=True)
        srl_annots = pd.read_csv(self.cfg.ds.trn_verb_ent_file)
        for k in srl_annots.columns:
            first_word = srl_annots.iloc[0][k]
            if isinstance(first_word, str) and first_word[0] == '[':
                srl_annots[k] = srl_annots[k].apply(
                    lambda x: ast.literal_eval(x))

        # arg_counter = Counter()
        # for r_arg in srl_annots.req_args:
            # arg_counter += Counter(r_arg)

        # arg_vocab = vocab.Vocab(arg_counter, specials_first=True)
        arg_vocab = create_vocab(srl_annots, 'req_args')
        arg_tag_vocab = create_vocab(srl_annots, 'tags')
        out_vocab = {'arg_vocab': arg_vocab, 'arg_tag_vocab': arg_tag_vocab}
        pickle.dump(out_vocab, file=open(self.cfg.ds.arg_vocab_file, 'wb'))
        return

    def glove_vocabs(self):
        # Load dictionaries
        self.comm.dic_anet = json.load(open(self.inp_dict_file))
        # Get detections to index
        self.comm.dtoi = {w: i+1 for w,
                          i in self.comm.dic_anet['wtod'].items()}
        self.comm.itod = {i: w for w, i in self.comm.dtoi.items()}
        self.comm.itow = self.comm.dic_anet['ix_to_word']
        self.comm.wtoi = {w: i for i, w in self.comm.itow.items()}

        self.comm.vocab_size = len(self.comm.itow) + 1
        self.comm.detect_size = len(self.comm.itod)

        # Load the glove vocab
        self.glove = vocab.GloVe(name='6B', dim=300)

        # get the glove vector for the vg detection cls
        # From Peter's repo
        obj_cls_file = self.cfg.ds.vg_class_file
        # index 0 is the background
        with open(obj_cls_file) as f:
            data = f.readlines()
            classes = ['__background__']
            classes.extend([i.strip() for i in data])

        # for VG classes
        # self.comm.vg_cls = classes

        # Extract glove vectors for the Visual Genome Classes
        # TODO: Cleaner implementation possible
        # TODO: Preproc only once
        glove_vg_cls = np.zeros((len(classes), 300))
        for i, w in enumerate(classes):
            split_word = w.replace(',', ' ').split(' ')
            vector = []
            for word in split_word:
                if word in self.glove.stoi:
                    vector.append(
                        self.glove.vectors[self.glove.stoi[word]].numpy())
                else:  # use a random vector instead
                    vector.append(2*np.random.rand(300) - 1)

            avg_vector = np.zeros((300))
            for v in vector:
                avg_vector += v

            glove_vg_cls[i] = avg_vector/len(vector)

        # category id to labels. +1 becuase 0 is the background label
        # Extract glove vectors for the 431 classes in AnetEntDataset
        # TODO: Cleaner Implementation
        # TODO: Preproc only once
        glove_clss = np.zeros((len(self.comm.itod)+1, 300))
        glove_clss[0] = 2*np.random.rand(300) - 1  # background
        for i, word in enumerate(self.comm.itod.values()):
            if word in self.glove.stoi:
                vector = self.glove.vectors[self.glove.stoi[word]]
            else:  # use a random vector instead
                vector = 2*np.random.rand(300) - 1
            glove_clss[i+1] = vector

        # Extract glove vectors for the words from the vocab
        # TODO: cleaner implementation
        # TODO: preproc only once
        glove_w = np.zeros((len(self.comm.wtoi)+1, 300))
        for i, word in enumerate(self.comm.wtoi.keys()):
            vector = np.zeros((300))
            count = 0
            for w in word.split(' '):
                count += 1
                if w in self.glove.stoi:
                    glove_vector = self.glove.vectors[self.glove.stoi[w]]
                    vector += glove_vector.numpy()
                else:  # use a random vector instead
                    random_vector = 2*np.random.rand(300) - 1
                    vector += random_vector
            glove_w[i+1] = vector / count

        out_dict = {
            'classes': classes,
            'glove_vg_cls': glove_vg_cls,
            'glove_clss': glove_clss,
            'glove_w': glove_w
        }
        pickle.dump(out_dict, open(self.cfg.ds.glove_stuff, 'wb'))


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/create_asrl_cfg.yml')))
    anet_csv = AnetCSV(cfg)

    anet_csv.create_csvs()

    anet_csv.post_proc('train')
    anet_csv.post_proc('val')

    anet_csv.post_proc_srl('train', 'val')
