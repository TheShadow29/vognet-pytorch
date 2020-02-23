"""
To create the 4-way dataset
Main motivation:
Currently, not sure if the models ground
based only on object name, or is it really
learning the roles of the visual elements
correctly.

Thus, we create 4-way dataset, for every
data which has S-V-O statistics, we generate
counterfactuals (not sure if this is
the correct name or not). For every image
containing say S1-V1-O1, present it with other
images with the characteristics S2-V1-O1,
S1-V2-O1, S1-V1-O2 as well. Some can be
reduced in case only S-V or O-V are present

More generally, we would like to create a
counterfactuals for anything that can
provide evidence.

Additionally, need to check
- [x] Location words shouldn't be present
- [x] Perform VERB lemmatization
- [x] Distinguish between what is groundable and
      what is not
- [x] Check the groundable verbs
"""
from pathlib import Path
import pandas as pd

from tqdm.auto import tqdm
from collections import Counter
import json
import copy
import ast
import numpy as np
from _init_stuff import CN, yaml

np.random.seed(seed=5)


def create_random_list(cfg, srl_annots, ann_row_idx):
    """
    Returns 4 random videos
    """
    srl_idxs_possible = np.array(srl_annots.index)

    vid_segs = srl_annots.vid_seg
    vid_seg = vid_segs.loc[ann_row_idx]
    srl_row = srl_annots.loc[ann_row_idx]

    req_cls_pats = srl_row.req_cls_pats
    req_cls_pats_mask = srl_row.req_cls_pats_mask
    args_to_use = set(['V', 'ARG0', 'ARG1', 'ARG2', 'ARGM-LOC'])

    arg_keys_vis_present = []
    arg_keys_lang_present = []
    for srl_arg, srl_arg_mask in zip(req_cls_pats, req_cls_pats_mask):
        arg_key = srl_arg[0]
        arg_keys_lang_present.append(arg_key)
        if arg_key == 'V' or arg_key in args_to_use:
            arg_keys_vis_present.append(arg_key)

    ds4_msk = {}
    inds_to_use = {}
    num_arg_keys_vis = len(arg_keys_vis_present)
    other_anns = np.random.choice(
        srl_idxs_possible, size=10 * num_arg_keys_vis,
        replace=False
    ).reshape(num_arg_keys_vis, 10)

    for aind, arg_key1 in enumerate(arg_keys_vis_present):
        in1 = other_anns[aind].tolist()
        assert len(in1) == 10

        set1 = set(in1)

        set_int = [s for s in set1 if
                   vid_segs.loc[s] != vid_seg]

        # TODO:
        # Make replace false, currently true
        # because some have low chances of
        # appearing
        assert len(set_int) > 0
        inds_to_use[arg_key1] = set_int
        ds4_msk[arg_key1] = 1
    return inds_to_use, ds4_msk


def create_similar_list(cfg, arg_dicts, srl_annots, ann_row_idx):
    """
    Does it for one row. Assumes annotations
    exists and can be retrieved via `self`.

    The logic:
    Each input idx has ARG0, V, ARG1 ...,
    (1) Pivot across one argument, say ARG0
    (2) Retrieve all other indices such that they
    have different ARG0, but same V, ARG1 ... (do
    each of them separately)
    (3) To retrieve those indices with V, ARG1 same
    we can just do intersection of the two sets

    To facilitate (2), we first create separate
    dictionaries for each V, ARG1 etc. and then
    just reference them via self.create_dicts
    """
    srl_idxs_possible = np.array(srl_annots.index)

    vid_segs = srl_annots.vid_seg
    vid_seg = vid_segs.loc[ann_row_idx]
    srl_row = srl_annots.loc[ann_row_idx]

    req_cls_pats = srl_row.req_cls_pats
    req_cls_pats_mask = srl_row.req_cls_pats_mask
    args_to_use = set(['V', 'ARG0', 'ARG1', 'ARG2', 'ARGM-LOC'])
    some_inds = {}
    arg_keys_vis_present = []
    arg_keys_lang_present = []
    for srl_arg, srl_arg_mask in zip(req_cls_pats, req_cls_pats_mask):
        arg_key = srl_arg[0]
        arg_keys_lang_present.append(arg_key)
        if arg_key == 'V' or arg_key in args_to_use:
            # If visually groundable
            # if (srl_arg_mask[1] == 1 or arg_key == 'V'):
            arg_keys_vis_present.append(arg_key)
            if arg_key in args_to_use:
                lemma_key = 'lemma_{}'.format(
                    arg_key.replace('-', '_').replace('V', 'verb'))
                lemma_arg = srl_row[lemma_key]
                if isinstance(lemma_arg, list):
                    assert all([le_arg in arg_dicts[arg_key]
                                for le_arg in lemma_arg])
                    if len(lemma_arg) >= 1:
                        le_arg = lemma_arg[0]
                    else:
                        le_arg = cfg.ds.none_word
                else:
                    le_arg = lemma_arg
                # srl_ind_list = copy.deepcopy(
                #     arg_dicts[arg_key][le_arg])
                # srl_ind_list = arg_dicts[arg_key][le_arg][:]
                srl_ind_list = arg_dicts[arg_key][le_arg][:]
                srl_ind_list.remove(ann_row_idx)
                if arg_key not in some_inds:
                    some_inds[arg_key] = []
                some_inds[arg_key] += srl_ind_list
            # # If not groundable but in args_to_use
            # else:
            #     pass
    num_arg_keys_vis = len(arg_keys_vis_present)
    other_anns = np.random.choice(
        srl_idxs_possible, size=10 * num_arg_keys_vis,
        replace=False
    ).reshape(num_arg_keys_vis, 10)

    inds_to_use = {}
    ds4_msk = {}
    for aind, arg_key1 in enumerate(arg_keys_vis_present):
        arg_key_to_use = [
            ak for ak in arg_keys_vis_present if ak != arg_key1]
        set1 = set(some_inds[arg_key_to_use[0]])

        set_int1 = set1.intersection(
            *[set(some_inds[ak]) for ak in arg_key_to_use[1:]])
        curr_set = set(some_inds[arg_key1])
        set_int2 = list(set_int1 - curr_set)

        set_int = [s for s in set_int2 if
                   vid_segs.loc[s] != vid_seg]

        # TODO:
        # Make replace false, currently true
        # because some have low chances of
        # appearing
        if len(set_int) == 0:
            # this means similar scenario not found
            # inds
            ds4_msk[arg_key1] = 0
            inds_to_use[arg_key1] = other_anns[aind].tolist()
            # inds_to_use[arg_key1] = [-1]
            # cfg.ouch += 1
            # print('ouch')
        else:
            ds4_msk[arg_key1] = 1
            inds_to_use[arg_key1] = np.random.choice(
                set_int, 10, replace=True).tolist()
            # cfg.yolo += 1
            # print('yolo')
    # inds_to_use_lens = [len(v) if v[0] != -1 else 0 for k,
    #                     v in inds_to_use.items()]
    # if sum(inds_to_use_lens) == 0:
    #     cfg.ouch2 += 1
    # else:
    #     cfg.yolo2 += 1

    return inds_to_use, ds4_msk


class AnetDSCreator:
    def __init__(self, cfg, tdir='.'):
        import spacy
        self.cfg = cfg
        self.tdir = Path(tdir)

        self.sp = spacy.load('en_core_web_sm')

        # Open required files
        self.open_req_files()

    def fix_via_ast(self, df):
        for k in df.columns:
            first_word = df.iloc[0][k]
            if isinstance(first_word, str) and (first_word[0] in '[{'):
                df[k] = df[k].apply(
                    lambda x: ast.literal_eval(x))
        return df

    def open_req_files(self):
        pass
        # verb_ent_file = self.tdir / Path(self.cfg.ds.verb_ent_file)
        # self.trn_ent_verb = pd.read_csv(verb_ent_file)

        # self.verb_lemma_lookup = {"'ve": "have", "'m": "am"}

        # self.lemmatized_verb_file = (self.tdir /
        #                              Path(self.cfg.ds.verb_lemma_file))
        # if self.lemmatized_verb_file.exists():
        #     self.all_verbs_df = pd.read_csv(self.lemmatized_verb_file)

        # self.lemmatized_verb_dict_file = (
        #     self.tdir /
        #     Path(self.cfg.ds.verb_lemma_dict_file)
        # )
        # if self.lemmatized_verb_dict_file.exists():
        #     self.verbs_lemma_dict = json.load(
        #         open(self.lemmatized_verb_dict_file))

        # self.srl_annots = copy.deepcopy(self.trn_ent_verb)
        # assert hasattr(self, 'srl_annots')
        # self.srl_annots = self.fix_via_ast(self.srl_annots)

    def get_stats(self, req_args):
        """
        Gets the counts for the argument types
        """
        c = Counter()
        if isinstance(req_args[0], list):
            for x in req_args:
                c += Counter(x)
        else:
            c = Counter(req_args)

        return c.most_common()

    def lemmatize_verbs(self):
        def get_lemma(word):
            if word in self.verb_lemma_lookup:
                return self.verb_lemma_lookup[word]
            words = self.sp(word)
            out_lemma = words[0].lemma_
            # if out_lemma in self.verb_lemma_lookup:
            # out_lemma = self.verb_lemma_lookup[out_lemma]
            return out_lemma
        self.all_verbs_df = self.trn_ent_verb[['verb']]
        tqdm.pandas()

        self.all_verbs_df['lemma_verb'] = self.all_verbs_df.verb.progress_apply(
            get_lemma
        )
        self.all_verbs_df.to_csv(
            self.lemmatized_verb_file, index=False, header=True)

        all_verbs_df_dropped_dupl = self.all_verbs_df.drop_duplicates()
        verbs_lemma_dict = {
            row.verb: row.lemma_verb for row_ind, row in
            all_verbs_df_dropped_dupl.iterrows()
        }
        json.dump(verbs_lemma_dict,
                  open(self.lemmatized_verb_dict_file, 'w'))
        return

    def create_all_similar_lists(self):
        self.create_similar_lists(split_type='train')
        self.create_similar_lists(split_type='valid')

    def create_similar_lists(self, split_type: str = 'train'):
        """
        need to check if only
        creating for the validation
        set would be enough or not.

        Basically, for each input,
        generates list of other inputs (idxs)
        which have same S,V,O (at least one is same)
        """
        if split_type == 'train':
            srl_annot_file = self.tdir / self.cfg.ds.trn_verb_ent_file
            ds4_dict_file = self.tdir / self.cfg.ds.trn_ds4_dicts
            ds4_ind_file = self.tdir / self.cfg.ds.trn_ds4_inds
        elif split_type == 'valid':
            srl_annot_file = self.tdir / self.cfg.ds.val_verb_ent_file
            ds4_dict_file = self.tdir / self.cfg.ds.val_ds4_dicts
            ds4_ind_file = self.tdir / self.cfg.ds.val_ds4_inds
        elif split_type == 'trn_val':
            srl_annot_file = self.tdir / self.cfg.ds.verb_ent_file
            ds4_dict_file = self.tdir / self.cfg.ds.ds4_dicts
            ds4_ind_file = self.tdir / self.cfg.ds.ds4_inds
        elif split_type == 'only_val':
            srl_annot_file = Path('./data/anet_verb/val_1_verb_ent_file.csv')
            ds4_dict_file = Path(
                './data/anet_verb/val_1_srl_args_dict_obj_to_ind.json'
            )
        else:
            raise NotImplementedError
        # elif split_type == 'test':
        #     srl_annot_file = self.tdir / self.cfg.ds.test_verb_ent_file
        #     ds4_dict_file = self.tdir / self.cfg.ds.test_ds4_dicts
        #     ds4_ind_file = self.tdir / self.cfg.ds.test_ds4_inds
        # elif split_type == 'val_test':
        #     # validation file with validation+test indices
        #     srl_annot_file = self.tdir / self.cfg.ds.test_verb_ent_file
        #     ds4_dict_file = self.tdir / self.cfg.ds.test_ds4_dicts
        #     ds4_ind_file = self.tdir / self.cfg.ds.test_ds4_inds
        # elif split_type == 'test_val':
        #     # test file with validation+test indices
        #     srl_annot_file = self.tdir / self.cfg.ds.test_verb_ent_file
        #     ds4_dict_file = self.tdir / self.cfg.ds.test_ds4_dicts
        #     ds4_ind_file = self.tdir / self.cfg.ds.test_ds4_inds
        # else:
        # raise NotImplementedError
        srl_annots = self.fix_via_ast(pd.read_csv(srl_annot_file))

        self.create_dicts_srl(srl_annots, ds4_dict_file)

        arg_dicts = json.load(open(ds4_dict_file))
        srl_annots_copy = copy.deepcopy(srl_annots)
        # inds_to_use_list = [self.create_similar_list(
        # row_ind) for row_ind in tqdm(range(len(self.srl_annots)))]
        inds_to_use_list = []
        ds4_msk = []
        rand_inds_to_use_list = []

        for row_ind in tqdm(range(len(srl_annots))):
            inds_to_use, ds4_msk_out = create_similar_list(
                self.cfg, arg_dicts, srl_annots, row_ind)
            ds4_msk.append(ds4_msk_out)

            inds_to_use_list.append(inds_to_use)

            rand_inds_to_use, _ = create_random_list(
                self.cfg, srl_annots, row_ind
            )
            rand_inds_to_use_list.append(rand_inds_to_use)

        srl_annots_copy['DS4_Inds'] = inds_to_use_list
        srl_annots_copy['ds4_msk'] = ds4_msk

        srl_annots_copy['RandDS4_Inds'] = rand_inds_to_use_list
        # srl_annots_copy = srl_annots_copy.iloc[ds4_msk]

        srl_annots_copy.to_csv(
            ds4_ind_file, index=False, header=True)
        # srl_annots_copy.to_csv(
        #     self.tdir/self.cfg.ds.ds4_inds, index=False, header=True)
        # for row_ind in range(len(self.srl_annots)):
        # inds_to_use = self.create_similar_list(row_ind)

    def create_dicts_srl(self, srl_annots, out_file):
        def default_dict_list(key_list, val, dct):
            for key in key_list:
                if key not in dct:
                    dct[key] = []
                dct[key].append(val)
            return dct

        # srl_annots = self.srl_annots

        # args_dict_out: Dict[str, Dict[obj_name, srl_indices]]
        # arg_dict_lemma: Dict[str, List[obj_name]]
        args_dict_out = {}
        args_to_use = ['ARG0', 'ARG1', 'ARG2', 'ARGM-LOC']
        for srl_arg in args_to_use:
            args_dict_out[srl_arg] = {}

        for row_ind, row in tqdm(srl_annots.iterrows(),
                                 total=len(srl_annots)):
            req_cls_pats = row.req_cls_pats
            req_cls_pats_mask = row.req_cls_pats_mask
            for srl_arg, srl_arg_mask in zip(req_cls_pats, req_cls_pats_mask):
                arg_key = srl_arg_mask[0]
                if arg_key in args_dict_out:
                    # The argument is groundable
                    if srl_arg_mask[1] == 1:
                        key_list = list(set(srl_arg[1]))
                        args_dict_out[arg_key] = default_dict_list(
                            key_list, row_ind, args_dict_out[arg_key])
                    else:
                        key_list = [self.cfg.ds.none_word]
                        args_dict_out[arg_key] = default_dict_list(
                            key_list, row_ind, args_dict_out[arg_key])

        args_dict_out['V'] = {k: list(v.index) for k,
                              v in srl_annots.groupby('lemma_verb')}
        json.dump(args_dict_out, open(out_file, 'w'))
        return args_dict_out


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/create_asrl_cfg.yml')))
    for split_type in ['valid', 'train', 'trn_val']:
        # for split_type in ['only_val', 'valid', 'train', 'trn_val']:
        # cfg.ouch = 0
        # cfg.yolo = 0

        # cfg.ouch2 = 0
        # cfg.yolo2 = 0

        anet_ds = AnetDSCreator(cfg)
        # anet_ds.lemmatize_verbs()
        # anet_ds.create_dicts_srl()
        anet_ds.create_similar_lists(split_type=split_type)

        break

        # anet_ds.create_similar_lists(split_type='trn_val')
        # anet_ds.create_similar_lists(split_type='train')
        # anet_ds.create_similar_lists(split_type='valid')
        # print(cfg.ouch, cfg.yolo, cfg.yolo+cfg.ouch)
        # print(cfg.ouch2, cfg.yolo2, cfg.yolo2+cfg.ouch2)
