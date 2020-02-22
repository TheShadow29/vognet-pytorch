"""
Perform semantic role labeling for input captions
"""
from allennlp.predictors.predictor import Predictor
import pandas as pd
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from yacs.config import CfgNode as CN
import time
import fire
import re
from typing import List, Dict, Any, Union

SRL_BERT = (
    "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")

srl_out_patt = re.compile(r'\[(.*?)\]')

Fpath = Union[Path, str]
Cft = CN


class SRL_DS:
    """
    A base class to perform semantic role labeling
    """

    def __init__(self, cfg: Cft, tdir: str = '.'):
        self.cfg = cfg
        self.tdir = Path(tdir)
        archive_path = SRL_BERT
        self.srl = Predictor.from_path(
            archive_path=archive_path,
            predictor_name='semantic-role-labeling',
            cuda_device=0
        )
        # self.srl = Predictor.from_path(
        # "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        self.name = self.__class__.__name__
        self.cache_dir = self.tdir / \
            Path(f'{self.cfg.misc.cache_dir}/{self.name}')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.out_file = (self.cache_dir / f'{self.cfg.ds.srl_bert}')
        self.after_init()

    def after_init(self):
        pass

    def get_annotations(self) -> pd.DataFrame:
        """
        Expected to read a file,
        Create a df with the columns:
        vid_id, seg_id, sentence
        """
        raise NotImplementedError

    def do_predictions(self):
        annot_df = self.get_annotations()
        sents = annot_df.to_dict('records')
        st_time = time.time()
        out_list = []
        tot_len = len(sents)
        try:
            for idx in tqdm(range(0, len(sents), 50)):
                out = self.srl.predict_batch_json(
                    sents[idx:min(idx+50, tot_len)])
                out_list += out
        except RuntimeError:
            pass
        finally:
            end_time = time.time()
            print(f'Took time {end_time-st_time}')
        pickle.dump(out_list, open(self.out_file, 'wb'))
        self.update_preds()

    def update_preds(self):
        preds = pickle.load(open(self.out_file, 'rb'))
        for pred in tqdm(preds):
            for verb in pred['verbs']:
                verb['req_pat'] = srl_out_patt.findall(verb['description'])
        pickle.dump(preds, open(self.out_file, 'wb'))


class SRL_Anet(SRL_DS):
    def after_init(self):
        """
        Assert files exists
        """
        # Assert Raw Caption Files exists
        self.trn_anet_cap_file = self.tdir / Path(self.cfg.ds.anet_cap_file)
        assert self.trn_anet_cap_file.exists()

    def get_annotations(self):
        trn_cap_data = json.load(open(self.trn_anet_cap_file))
        trn_vid_list = list(trn_cap_data.keys())
        out_dict_list = []
        for trn_vid_name in tqdm(trn_vid_list):
            trn_vid_segs = trn_cap_data[trn_vid_name]
            num_segs = len(trn_vid_segs['timestamps'])
            for seg in range(num_segs):
                out_dict = {
                    'time_stamp': trn_vid_segs['timestamps'][seg],
                    'vid': trn_vid_name,
                    'vid_seg': f'{trn_vid_name}_segment_{seg:02d}',
                    'segment': seg,
                    'sentence': trn_vid_segs['sentences'][seg]
                }
                out_dict_list.append(out_dict)
        out_df = pd.DataFrame(out_dict_list)
        out_df.to_csv(
            (
                self.cache_dir /
                f'{self.cfg.ds.srl_caps}'
            ),
            header=True, index=False
        )
        return out_df


def main():
    cfg_file = './configs/create_asrl_cfg.yml'
    cfg = CN(yaml.safe_load(open(cfg_file)))
    print(cfg)
    srl_ds = SRL_Anet(cfg)
    srl_ds.do_predictions()


if __name__ == '__main__':
    main()
    # fire.Fire(main)
