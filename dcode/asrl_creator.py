"""
Create ASRL dataset
"""
from yacs.config import CfgNode as CN
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pickle
# import re
from spacy.gold import align
from typing import List
from collections import Counter
import yaml


def get_corr_ind(tok1: List[str], tok2: List[str],
                 tok1_idx: List[List[int]]) -> List[List[int]]:
    """
    Aligns two different tokenizations
    and outputs the tok2_idx.
    tok1: tokenized sents via method1
    tok2: tokenized sents via method2
    tok1_idx: indices of tok1
    output: tok2_idx: indices of tok2
    based on tok1_idx
    """
    cost, a2b, b2a, a2b_multi, b2a_multi = align(tok1, tok2)
    # If aligned no pains
    # can directly return tok1_idx
    if cost == 0:
        return tok1_idx

    # Else create tok2_idx
    tok2_idx = []
    for t1_idx in tok1_idx:
        t2_idx = []
        for t in t1_idx:
            # If the tok1_idx corresponds
            # to one single token of tok2
            # just use that
            if a2b[t] != -1:
                t2_idx.append(a2b[t])
            # else use multi outputs
            else:
                # hacky implementation
                # Basically, if the previous word is aligned,
                # and the next word as well, assign current words
                # to the difference of the two
                if t != len(tok1) - 1:
                    if a2b[t-1] != -1 and a2b[t+1] != -1:
                        t2_idx.append(
                            [x for x in range(a2b[t-1] + 1, a2b[t+1])])
                elif a2b[t-1] != -1:
                    t2_idx.append(
                        [x for x in range(a2b[t-1]+1, len(tok2))])
                else:
                    # Currently seems to work,
                    # set_trace to see when it doesn't work
                    import pdb
                    pdb.set_trace()
                    pass
        tok2_idx.append(t2_idx)

    return tok2_idx


def get_pat_from_tags(tags: List[str],
                      given_req_pat: List[str],
                      word_list: List[str]) -> List[List]:
    """
    tags is a list ['O', 'B-ARG0'...]
    need to retrieve 'ARG0': w1...wk

    given_req_pat is a list ['ARG0: w1 .. wk', ..]

    word_list is tokenized word list

    output: [['ARG0': [indices]], ...]

    For Sanity, don't try to understand this function.
    Assume, Black Magic
    """
    # Logic is a bit complicated?
    # if we see 'O' or 'B', then finish previously started
    # stuff. assert only one has started and append it to the list
    # If it is 'B', start collecting in the dictionary

    req_pats = []
    req_pat = {}
    started = False
    curr_tag = ''
    for word_ind, tag in enumerate(tags):
        if (tag == 'O' or (tag[0] == 'B')):
            if started:
                started = False
                keys = list(req_pat.keys())
                assert len(keys) == 1
                key = keys[0]
                req_pats.append([key, req_pat[key]])
                req_pat = {}

        if tag != 'O':
            if tag[0] == 'B':
                curr_tag = tag[2:]
                assert curr_tag not in req_pat
                req_pat[curr_tag] = [word_ind]
                started = True
            elif tag[0] == 'I':
                if not started:
                    continue
                # TODO: NEEDS VERIFICATION
                # MIGHT WANT TO REMOVE SUCH CASES
                # ALTOGETHER
                # Currently, skip checking
                # if not tag[2:] == curr_tag:
                # assert tag[2:] == curr_tag[2:]
                req_pat[curr_tag].append(word_ind)
        if word_ind == len(tags)-1:
            if started:
                started = False
                keys = list(req_pat.keys())
                assert len(keys) == 1
                key = keys[0]
                req_pats.append([key, req_pat[key]])
                req_pat = {}

    # Assert this pattern is correct by comparing to gt
    # ASSERTION PASSED. YAY
    assert len(given_req_pat) == len(req_pats)
    for g, r in zip(given_req_pat, req_pats):
        g_arg, g_name = g.split(':', 1)
        r_arg, r_name = r
        obtained_words = ' '.join([word_list[ix] for ix in r_name])
        assert obtained_words.strip() == g_name.strip()

    return req_pats


def get_clss_from_pats(
        req_pats: List[List],
        idx2: List[List],
        cls_idx1: List[List],
        word_list=None,
        gvd_info=None) -> List[List]:
    """
    Corresponds the words to the classes.
    req_pats: [['ARG0': [1,2]], ...]
    idx2: Returned from alignment output
    of the type [[W1], [WK, WL],...]
    cls_idx1: class names based on idx1 (not input here)
    But idx1 and idx2 have one-to-one mapping so
    just need the indice of idx2 to get the corresponding
    class name.
    outputs: same as req_pats, but if the word has a class name
    use that, if not class name use the actual word

    May God give me courage to parse my own code!
    """
    def list_to_tuple(lst):
        ix2_t = []
        for l in lst:
            if isinstance(l, list):
                ix2_t += [tuple(l)]
            else:
                ix2_t += [tuple([l])]
        return tuple(ix2_t)

    out_req_pats = []
    out_req_pats_mask = []

    idx2_dict = {}

    for idx2_ind, ix2 in enumerate(idx2):
        # if isinstance(ix2, list):
        ix2_t = list_to_tuple(ix2)
        if ix2_t in idx2_dict:
            idx2_dict[ix2_t].append(idx2_ind)
        else:
            idx2_dict[ix2_t] = [idx2_ind]
    # Logic is slightly complicated.
    # Basically, see if there is any intersection
    # between the idx2 and the req_pat indices
    for r_arg, r_name in req_pats:
        guess = False
        for idx2_ind, ix2 in enumerate(idx2):
            ix2_set = set()
            for ix22 in ix2:
                if isinstance(ix22, list):
                    ix2_set = ix2_set.union(set(ix22))
                else:
                    ix2_set = ix2_set.union(set([ix22]))
            if set(r_name).intersection(ix2_set):
                # assert not guess
                guess = True
                cls_name = cls_idx1[idx2_ind]
                out_req_pats.append((r_arg, cls_name))
                ix2_t = list_to_tuple(ix2)
                idx2_list = idx2_dict[ix2_t]
                out_req_pats_mask.append((r_arg, 1, idx2_list))
                break
        if not guess:
            if word_list is None:
                out_req_pats.append((r_arg, r_name))
            else:
                r_name_word = [word_list[ix] for ix in r_name]
                out_req_pats.append((r_arg, r_name_word))
            out_req_pats_mask.append((r_arg, 0, [0]))
    assert len(out_req_pats) == len(req_pats)
    assert len(out_req_pats_mask) == len(out_req_pats)
    return out_req_pats, out_req_pats_mask


class BaseVis:
    def get_svo(self, inp):
        """
        Convinience function to get the
        subject, verb, objects
        """
        cls_pats = inp.req_cls_pats
        cls_pats_mask = inp.req_cls_pats_mask
        s = None
        o = None
        v = None
        for x, y in zip(cls_pats, cls_pats_mask):
            if x[0] == 'ARG0':
                if y[1] == 1:
                    s = x[1][0]
                else:
                    s = 'NOPE'
            if x[0] == 'V':
                v = x[1][0]
            if x[0] == 'ARG1':
                if y[1] == 1:
                    o = x[1][0]
                else:
                    o = 'NOPE'
        assert s is not None
        assert o is not None
        assert v is not None
        return f'{s}-{v}-{o}'

    def get_srl_stats(self):
        """
        Gets the counts for the argument types
        """
        req_args = self.trn_ent_verb_df.req_args
        c = Counter()
        for x in req_args:
            c += Counter(x)
        return c.most_common()

    def get_svo_stats(self):
        """
        Gets the svo statistics
        """
        svo_list = ['ARG0', 'V', 'ARG1']
        self.trn_ent_verb_df_svo = self.trn_ent_verb_df[
            self.trn_ent_verb_df.req_args.apply(
                lambda x: len(
                    set(x).intersection(set(svo_list))
                ) == len(svo_list))]
        self.svo_df = self.trn_ent_verb_df_svo.apply(self.get_svo, axis=1)

        svo_s = self.svo_df.apply(lambda x: x.split('-')[0])
        svo_v = self.svo_df.apply(lambda x: x.split('-')[1])
        svo_o = self.svo_df.apply(lambda x: x.split('-')[2])
        svo_sv = self.svo_df.apply(lambda x: tuple(x.split('-')[0: 2]))
        svo_vo = self.svo_df.apply(lambda x: tuple(x.split('-')[1:]))
        svo_so = self.svo_df.apply(lambda x: tuple(x.split('-')[0: 3: 2]))

        self.svo_stats = [svo_s, svo_v, svo_o,
                          svo_sv, svo_vo, svo_so, self.svo_df]
        return [len(x.unique()) for x in self.svo_stats]


class AnetVis(BaseVis):
    """
    Helper class for visualization
    """

    def __init__(self, cfg, tdir='.'):
        self.cfg = cfg
        self.tdir = Path(tdir)

        self.cache_dir = (
            self.tdir / Path(f'{self.cfg.misc.cache_dir}/SRL_Anet'))

        # Load the annotations
        self.load_annots_gvd()
        self.open_req_files()

        # set include/exclude sets
        self.exclude_verb_set = set(self.cfg.ds.exclude_verb_set)
        self.include_arg_list = self.cfg.ds.include_srl_args

    def lemmatize_verbs(self):
        import spacy
        self.sp = spacy.load('en_core_web_sm')

        self.verb_lemma_lookup = {"'ve": "have", "'m": "am"}

        def get_lemma(word):
            if word in self.verb_lemma_lookup:
                return self.verb_lemma_lookup[word]
            words = self.sp(word)
            out_lemma = words[0].lemma_
            # if out_lemma in self.verb_lemma_lookup:
            # out_lemma = self.verb_lemma_lookup[out_lemma]
            return out_lemma
        verb_list = [vb['verb']
                     for sent in self.srl_bert for vb in sent['verbs']]
        verb_list_uniq = list(set(verb_list))

        verbs_lemma_dict = {
            v: get_lemma(v) for v in verb_list_uniq
        }

        json.dump(verbs_lemma_dict,
                  open(self.verb_lemma_dict_file, 'w'))
        return

    def save_trn_ent_file(self):
        save_fp = self.tdir / Path(self.cfg.ds.verb_ent_file)
        self.trn_ent_verb_df.to_csv(save_fp, header=True,
                                    index=False)

    def open_req_files(self):
        cap_annots_file = self.cache_dir / f'{self.cfg.ds.srl_caps}'
        assert cap_annots_file.exists() and cap_annots_file.suffix == '.csv'
        self.cap_annots = pd.read_csv(cap_annots_file)
        self.cap_annots_dict = {k: v for k,
                                v in self.cap_annots.groupby('vid_seg')}

        srl_bert_file = self.cache_dir / f'{self.cfg.ds.srl_bert}'
        assert srl_bert_file.exists() and srl_bert_file.suffix == '.pkl'
        self.srl_bert = pickle.load(open(srl_bert_file, 'rb'))

        self.trn_anet_gvd_ent_orig_file = self.tdir / \
            Path(self.cfg.ds.orig_anet_ent_clss)
        assert self.trn_anet_gvd_ent_orig_file.exists()
        self.trn_anet_gvd_ent_orig_data = json.load(
            open(self.trn_anet_gvd_ent_orig_file))

        self.verb_lemma_dict_file = self.tdir / \
            Path(self.cfg.ds.verb_lemma_dict_file)
        if not self.verb_lemma_dict_file.exists():
            self.lemmatize_verbs()
        self.verb_lemma_dict = json.load(open(self.verb_lemma_dict_file))

    def load_srl_annots(self):
        def get_verb(vid_seg):
            cap_annot_index = self.cap_annots_dict[vid_seg].index[0]
            srl_info = self.srl_bert[cap_annot_index]
            return srl_info
        assert isinstance(self.trn_ent_df, pd.DataFrame)
        self.trn_dict_vid_seg = list(self.trn_ent_df.groupby('vid_seg'))

        exclude_set = self.exclude_verb_set
        args_to_use = self.include_arg_list
        args_to_use_set = set(self.include_arg_list)
        out_dict_list = []

        for k, v in tqdm(self.trn_dict_vid_seg):
            vid, seg = k.split('_segment_')
            seg = str(int(seg))
            srl_info = get_verb(k)
            sent = ' '.join(srl_info['words'])
            gvd_info = (
                self.trn_anet_gvd_ent_orig_data['annotations'][vid]['segments'][seg])

            # Do some aligning magic
            srl_idx = get_corr_ind(gvd_info['tokens'], srl_info['words'],
                                   gvd_info['process_idx'])

            for verb in srl_info['verbs']:
                if set(verb['tags']) == {'O'}:
                    continue
                lemma_verb = self.verb_lemma_dict[verb['verb']]

                if lemma_verb in exclude_set:
                    continue

                given_req_pat = verb['req_pat']
                given_req_pat = [r for r in given_req_pat if ':' in r]
                req_pat = get_pat_from_tags(
                    verb['tags'], given_req_pat, srl_info['words'])
                out_dict = {}
                out_dict['vid_seg'] = k
                out_dict['req_pat_ix'] = req_pat
                out_dict['words'] = srl_info['words']
                out_dict['sent'] = sent
                out_dict['verb'] = verb['verb']
                out_dict['lemma_verb'] = lemma_verb

                req_pat_words = [(r_arg, ' '.join([srl_info['words'][ix]
                                                   for ix in r_name]))
                                 for r_arg, r_name in req_pat]
                out_dict['req_pat'] = req_pat_words

                out_dict['req_args'] = [r_arg for r_arg, r_name in req_pat]
                out_dict['req_aname'] = [
                    r_name for r_arg, r_name in req_pat_words]
                out_dict['tags'] = verb['tags']
                req_cls_pats, req_cls_pats_mask = get_clss_from_pats(
                    req_pat, srl_idx, gvd_info['process_clss'],
                    srl_info['words'], gvd_info)
                out_dict['req_cls_pats'], out_dict['req_cls_pats_mask'] = (
                    req_cls_pats, req_cls_pats_mask)
                out_dict['process_idx2'] = srl_idx
                out_dict['process_clss'] = gvd_info['process_clss']

                if len(
                        set(
                            out_dict['req_args']
                        ).intersection(args_to_use_set)) == 0:
                    continue
                if 'V' not in out_dict['req_args']:
                    continue

                to_append = {k: [] for k in args_to_use}
                for srl_arg, srl_arg_mask in zip(req_cls_pats,
                                                 req_cls_pats_mask):
                    # The argument is groundable
                    if srl_arg_mask[1] == 1:
                        key_list = list(set(srl_arg[1]))
                        arg_key = srl_arg_mask[0]
                        if arg_key in args_to_use:
                            to_append[arg_key] = key_list
                for k1 in args_to_use:
                    lemma_key = 'lemma_{}'.format(k1.replace('-', '_'))
                    out_dict[lemma_key] = to_append[k1]
                out_dict_list.append(out_dict)

        self.trn_ent_verb_df = pd.DataFrame(out_dict_list)

        return

    def load_annots_gvd(self):
        """
        Load annots from GVD prepared files
        Better for consistency
        """
        # Assert GVD trn file exists
        trn_anet_gvd_ent_file = self.tdir / \
            Path(self.cfg.ds.anet_ent_annot_file)
        assert trn_anet_gvd_ent_file.exists()
        trn_anet_gvd_ent_data = json.load(open(trn_anet_gvd_ent_file))

        trn_anet_gvd_ent_data = trn_anet_gvd_ent_data

        trn_vid_seg_list = []
        trn_vid_list = list(trn_anet_gvd_ent_data.keys())

        # Assert Raw Caption Files exists
        trn_anet_cap_file = self.tdir / Path(self.cfg.ds.anet_cap_file)
        assert trn_anet_cap_file.exists()

        trn_cap_data = json.load(open(trn_anet_cap_file))

        for trn_vid_name in tqdm(trn_vid_list):
            trn_vid_segs = trn_anet_gvd_ent_data[trn_vid_name]['segments']
            seg_nums = sorted(list(trn_vid_segs.keys()), key=lambda x: int(x))
            trn_cap = trn_cap_data[trn_vid_name]

            for seg_num in seg_nums:
                obj_dicts = trn_vid_segs[seg_num]
                time_stamp = trn_cap['timestamps'][int(seg_num)]
                sent = trn_cap['sentences'][int(seg_num)]
                sent_toks = obj_dicts['caption']
                for as_ind, box in enumerate(obj_dicts['bbox']):
                    tok_idx = obj_dicts['idx'][as_ind]
                    toks = [sent_toks[t] for t in tok_idx]
                    out_dict = {
                        'time_stamp': time_stamp,
                        'sent': sent,
                        'sent_toks': sent_toks,
                        'bbox': box,
                        'frm_idx': obj_dicts['frm_idx'][as_ind],
                        'clss': obj_dicts['clss'][as_ind],
                        'tok_idx': tok_idx,
                        'toks_grnd': toks,
                        'segment': seg_num,
                        'vid': trn_vid_name,
                        'vid_seg': f'{trn_vid_name}_segment_{int(seg_num):02d}'
                    }
                    trn_vid_seg_list.append(out_dict)
        self.trn_vid_seg_list = trn_vid_seg_list
        self.trn_ent_df = pd.DataFrame(self.trn_vid_seg_list)


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/create_asrl_cfg.yml')))
    anet_vis = AnetVis(cfg)
    anet_vis.load_srl_annots()
    anet_vis.save_trn_ent_file()
