"""
Gives the dataset statistics
in form of tables.
Copy-paste to Excel for visualization
"""
from yacs.config import CfgNode as CN
import yaml
from asrl_creator import Anet_SRL_Create
from pathlib import Path
import pandas as pd
import ast
from typing import Dict, List, Tuple
from collections import Counter
import altair as alt


class AnetSRL_Vis(object):
    def __init__(self, cfg, do_vis=True):
        self.cfg = cfg
        self.open_req_files()
        self.vis = do_vis

    def fix_via_ast(self, df):
        for k in df.columns:
            first_word = df.iloc[0][k]
            if isinstance(first_word, str) and (first_word[0] in '[{'):
                df[k] = df[k].apply(
                    lambda x: ast.literal_eval(x))
        return df

    def open_req_files(self):
        trn_asrl_file = self.cfg.ds.trn_ds4_inds
        val_asrl_file = self.cfg.ds.val_ds4_inds

        self.trn_srl_annots = self.fix_via_ast(pd.read_csv(trn_asrl_file))
        self.val_srl_annots = self.fix_via_ast(pd.read_csv(val_asrl_file))

    def print_most_common_table(self, most_comm: List[Tuple]):
        """
        Prints most common output from a Counter in the
        form of a table for easy copy/pasting
        """
        patt = '{}, {}\n'
        out_str = ''
        for it in most_comm:
            out_str += patt.format(*it)
        print(out_str)
        return

    def visualize_df(self, df: pd.DataFrame,
                     x_name: str, y_name: str):
        bars = alt.Chart(df).mark_bar(
            cornerRadiusBottomRight=3,
            cornerRadiusTopRight=3,
        ).encode(
            x=alt.X(x_name, axis=alt.Axis(title="")),
            y=alt.Y(y_name, axis=alt.Axis(title=""),
                    sort='-x'),
            color=alt.value('#6495ED')
        )
        text = bars.mark_text(
            align='left',
            baseline='middle',
            dx=3  # Nudges text to right so it doesn't appear on top of the bar
        ).encode(
            text='Count:Q'
        )

        return (bars + text).properties(height=500)

    def get_num_vids(self):
        """
        Input dictionary with train and validation df
        """
        nvids = {}
        nvids['train'] = len(self.trn_srl_annots.vid_seg.unique())
        nvids['valid'] = len(
            self.val_srl_annots[
                self.val_srl_annots.vt_split == 'val'
            ].vid_seg.unique()
        )
        nvids['test'] = len(
            self.val_srl_annots[
                self.val_srl_annots.vt_split == 'test'
            ].vid_seg.unique()
        )
        return nvids

    def get_num_noun_phrase(self):
        """
        Return number of noun-phrases for
        each SRL
        """
        # req_cls_pats_mask: [['ArgX', 1/0, box_num]]
        # get only the argument name and count
        arg_counts = self.trn_srl_annots.req_cls_pats_mask.apply(
            lambda x: [y[0] for y in x]
        )
        return Counter([ac for acs in arg_counts for ac in acs])

    def get_num_phrase_with_box(self):
        # req_cls_pats_mask: [['ArgX', 1/0, box_num]]
        # get only the argument name and count
        arg_counts = self.trn_srl_annots.req_cls_pats_mask.apply(
            lambda x: [y[0] for y in x if y[1] == 1]
        )
        return Counter([ac for acs in arg_counts for ac in acs])

    def get_num_srl_structures(self):
        arg_struct_counts = self.trn_srl_annots.req_args.apply(
            lambda x: '-'.join(x)
        )
        return Counter(list(arg_struct_counts)).most_common(20)

    def get_num_lemma(self, arg_list):
        lemma_counts = {}
        col_set = set(self.trn_srl_annots.columns)
        for agl in arg_list:
            if agl != 'verb':
                lemma_key = f'lemma_{agl}'
                assert lemma_key in col_set
                lemma_counts[lemma_key] = Counter(
                    list(
                        self.trn_srl_annots[lemma_key].apply(
                            lambda x: x[0] if len(x) > 0 else ''
                        )
                    )
                )
            else:
                lemma_key = 'lemma_verb'
                lemma_counts[lemma_key] = Counter(
                    list(
                        self.trn_srl_annots[lemma_key]
                    )
                )
        return lemma_counts

    def get_num_q_per_vid(self):
        num_q_per_vid = (
            len(self.trn_srl_annots) /
            len(self.trn_srl_annots.vid_seg.unique())
        )

        num_srl_per_q = self.trn_srl_annots.req_args.apply(
            lambda x: len(x)).mean()

        num_w_per_q = self.trn_srl_annots.req_pat_ix.apply(
            lambda x: sum([len(y[1]) for y in x])).mean()

        return num_q_per_vid, num_srl_per_q, num_w_per_q

    def print_all_stats(self):
        vis_list = []
        nvid = self.get_num_vids()
        print("Number of videos in Train/Valid/Test: "
              f"{nvid['train']}, {nvid['valid']}, {nvid['test']}")

        num_q_per_vid, num_srl_per_q, num_w_per_q = self.get_num_q_per_vid()
        print(f"Number of Queries per Video is {num_q_per_vid}")
        print(f"Number of Queries per Video is {num_srl_per_q}")
        print(f"Number of Queries per Video is {num_w_per_q}")

        num_noun_phrases_for_srl = self.get_num_noun_phrase().most_common(n=20)
        num_np_srl = pd.DataFrame.from_records(
            data=num_noun_phrases_for_srl,
            columns=['Arg', 'Count']
        )
        if self.vis:
            vis_list.append(
                self.visualize_df(num_np_srl, x_name='Count:Q', y_name='Arg:O')
            )
        print('Noun Phrases Count')
        print(num_np_srl.to_csv(index=False))

        num_noun_phrases_with_box_for_srl = self.get_num_phrase_with_box()

        num_grnd_np_srl = pd.DataFrame.from_records(
            data=num_noun_phrases_with_box_for_srl.most_common(n=20),
            columns=['Arg', 'Count']
        )
        if self.vis:
            vis_list.append(
                self.visualize_df(
                    num_grnd_np_srl, x_name='Count:Q', y_name='Arg:O')
            )
        print('Groundable Noun Phrase Count')
        print(num_grnd_np_srl.to_csv(index=False))

        num_srl_struct = self.get_num_srl_structures()
        num_srl_struct_df = pd.DataFrame.from_records(
            data=num_srl_struct,
            columns=['Arg', 'Count']
        )
        if self.vis:
            vis_list.append(
                self.visualize_df(num_srl_struct_df,
                                  x_name='Count:Q', y_name='Arg:O')
            )
        print('SRL Structures Frequency')
        print(num_srl_struct_df.to_csv(index=False))

        arg_list = ['verb', 'ARG0', 'ARG1', 'ARG2', 'ARGM_LOC']
        lemma_counts = self.get_num_lemma(arg_list)
        min_t = 20
        num_lemma_args = {
            k: len([z for z in v.most_common() if z[1] > min_t])
            for k, v in lemma_counts.items()
        }
        print(f'Lemmatized Counts for each lemma: {num_lemma_args}')

        df_dict = {
            k: pd.DataFrame.from_records(
                data=v.most_common(21),
                columns=['String', 'Count']
            )
            for k, v in lemma_counts.items()
        }

        for k in df_dict:
            print(f'Most Frequent Lemmas for {k}')
            print(df_dict[k].to_csv(index=False))

        return lemma_counts
        # return vis_list


if __name__ == '__main__':
    cfg = CN(yaml.safe_load(open('./configs/anet_srl_cfg.yml')))
    asrl_vis = AnetSRL_Vis(cfg)
    asrl_vis.print_all_stats()
