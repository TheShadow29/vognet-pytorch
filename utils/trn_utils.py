"""
Utility functions
"""
from typing import Dict, List, Optional, Union, Any, Callable
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from pathlib import Path
import sys
import re
import numpy as np
from collections import Counter
from tqdm import tqdm
import time
import shutil
import json
from fastprogress.fastprogress import master_bar, progress_bar
import logging
import pickle
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist
from torch.distributed import ReduceOp
from yacs.config import CfgNode as CN

# from maskrcnn_benchmark.utils.checkpoint import load_state_dict


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def reduce_dict(input_dict, average=False):
    """
    Args:
    input_dict (dict): all the values will be reduced
    average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            dist.reduce(values, dst=0)
            # if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            # values /= world_size
        if average:
            values /= world_size
            reduced_dict = {
                k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_dict_corr(input_dict, nums):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    new_inp_dict = {k: v*nums for k, v in input_dict.items()}
    out_dict = reduce_dict(new_inp_dict)
    dist.reduce(nums, dst=0)
    if not is_main_process():
        return out_dict
    out_dict_avg = {k: v / nums.item() for k, v in out_dict.items()}
    return out_dict_avg


def exec_func_if_main_proc(func: Callable):
    def wrapper(*args, **kwargs):
        if is_main_process():
            func(*args, **kwargs)
    return wrapper


@dataclass
class DataWrap:
    path: Union[str, Path]
    train_dl: DataLoader
    valid_dl: DataLoader
    test_dl: Optional[Union[DataLoader, Dict]] = None


class NewDistributedSampler(DistributedSampler):
    """
    Same as default distributed sampler of pytorch
    Just has another argument for shuffle
    Allows distributed in validation/testing as well
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


def make_data_sampler(dataset: Dataset, shuffle: bool,
                      distributed: bool) -> Sampler:
    if distributed:
        return NewDistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def get_dataloader(cfg, dataset: Dataset, is_train: bool,
                   collate_fn) -> DataLoader:
    is_distributed = cfg.do_dist
    images_per_gpu = cfg.train.bs if is_train else cfg.train.bsv
    nw = cfg.train.nw if is_train else cfg.train.nwv
    if is_distributed:
        # DistributedDataParallel
        batch_size = images_per_gpu
        num_workers = nw
    elif cfg.do_dp:
        # DataParallel
        batch_size = images_per_gpu * cfg.num_gpus
        num_workers = nw * cfg.num_gpus
    else:
        batch_size = images_per_gpu
        num_workers = nw

    if is_train:
        shuffle = True
    else:
        shuffle = False if not is_distributed else True
        # shuffle = False

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    # if ((cfg.ds.ds4_type == 'sigmoid' or cfg.ds.ds4_type == 'sigmoid_single_q')
    #         and cfg.ds.ds4_screen == 'screen_sep'):
    #     collator = BatchCollatorDS4(cfg)
    # else:
    collator = collate_fn(cfg)

    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, drop_last=is_train,
                      num_workers=num_workers,
                      collate_fn=collator)


class SmoothenValue():
    """
    Create a smooth moving average for a value(loss, etc) using `beta`.
    Adapted from fastai(https://github.com/fastai/fastai)
    """

    def __init__(self, beta: float):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.smooth = 0

    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * \
            self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


class SmoothenDict:
    "Converts list to dicts"

    def __init__(self, keys: List[str], val: int):
        self.keys = keys
        self.smooth_vals = {k: SmoothenValue(val) for k in keys}

    def add_value(self, val: Dict[str, torch.tensor]):
        for k in self.keys:
            self.smooth_vals[k].add_value(val[k].detach())

    @property
    def smooth(self):
        return {k: self.smooth_vals[k].smooth for k in self.keys}

    @property
    def smooth1(self):
        return self.smooth_vals[self.keys[0]].smooth

    def tostring(self):
        out_str = ''
        for k in self.keys:
            out_str += f'{k}: {self.smooth_vals[k].smooth.item():.4f} '
        out_str += '\n'
        return out_str


def compute_avg(inp: List, nums: torch.tensor) -> float:
    "Computes average given list of torch.tensor and numbers corresponding to them"
    return (torch.stack(inp) * nums).sum() / nums.sum()


def compute_avg_dict(inp: Dict[str, List],
                     nums: torch.tensor) -> Dict[str, float]:
    "Takes dict as input"
    out_dict = {}
    for k in inp:
        out_dict[k] = compute_avg(inp[k], nums)

    return out_dict


def good_format_stats(names, stats) -> str:
    "Format stats before printing."
    str_stats = []
    for name, stat in zip(names, stats):
        t = str(stat) if isinstance(stat, int) else f'{stat.item():.4f}'
        t += ' ' * (len(name) - len(t))
        str_stats.append(t)
    return '  '.join(str_stats)


@dataclass
class Learner:
    uid: str
    data: DataWrap
    mdl: nn.Module
    loss_fn: nn.Module
    cfg: Dict
    eval_fn: nn.Module
    opt_fn: Callable
    device: torch.device = torch.device('cuda')

    def __post_init__(self):
        "Setup log file, load model if required"

        # Get rank
        self.rank = get_rank()

        self.init_log_dirs()

        self.prepare_log_keys()

        self.prepare_log_file()

        self.logger = self.init_logger()

        # Set the number of iterations, epochs, best_met to 0.
        # Updated in loading if required
        self.num_it = 0
        self.num_epoch = 0
        self.best_met = 0

        # Resume if given a path
        if self.cfg.train['resume']:
            self.load_model_dict(
                resume_path=self.cfg.train['resume_path'],
                load_opt=self.cfg.train['load_opt'])

        # self.writer.add_text(tag='cfg', text_string=json.dumps(self.cfg),
        # global_step=self.num_epoch)

    def init_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if not is_main_process():
            return logger
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(str(self.extra_logger_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # logging.basicConfig(
        #     filename=self.extra_logger_file,
        #     filemode='a',
        #     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        #     datefmt='%m/%d/%Y %H:%M:%S',
        #     level=logging.INFO
        # )
        return logger

    def init_log_dirs(self):
        """
        Convenience function to create the following:
        1. Log dir to store log file in txt format
        2. Extra Log dir to store the logger output
        3. Tb log dir to store tensorboard files
        4. Model dir to save the model files
        5. Predictions dir to store the predictions of the saved model
        6. [Optional] Can add some 3rd party logger
        """
        # Saves the text logs
        self.txt_log_file = Path(
            self.data.path) / 'txt_logs' / f'{self.uid}.txt'

        # Saves the output of self.logger
        self.extra_logger_file = Path(
            self.data.path) / 'ext_logs' / f'{self.uid}.txt'

        # Saves SummaryWriter outputs
        self.tb_log_dir = Path(self.data.path) / 'tb_logs' / f'{self.uid}'

        # Saves the trained model
        self.model_file = Path(self.data.path) / 'models' / f'{self.uid}.pth'

        # Saves the output predictions
        self.predictions_dir = Path(
            self.data.path) / 'predictions' / f'{self.uid}'

        self.create_log_dirs()

    @exec_func_if_main_proc
    def create_log_dirs(self):
        """
        Creates the directories initialized in init_log_dirs
        """
        self.txt_log_file.parent.mkdir(exist_ok=True, parents=True)
        self.extra_logger_file.parent.mkdir(exist_ok=True)
        self.tb_log_dir.mkdir(exist_ok=True, parents=True)
        self.model_file.parent.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True, parents=True)

    def prepare_log_keys(self):
        """
        Creates the relevant keys to be logged.
        Mainly used by the txt logger to output in a good format
        """
        def _prepare_log_keys(keys_list: List[List[str]],
                              prefix: List[str]) -> List[str]:
            """
            Convenience function to create log keys
            keys_list: List[loss_keys, met_keys]
            prefix: List['trn', 'val']
            """
            log_keys = []
            for keys in keys_list:
                for key in keys:
                    log_keys += [f'{p}_{key}' for p in prefix]
            return log_keys

        self.loss_keys = self.loss_fn.loss_keys
        self.met_keys = self.eval_fn.met_keys

        # When writing Training and Validation together
        self.trn_met = False
        self.log_keys = ['epochs'] + _prepare_log_keys(
            [self.loss_keys],
            ['trn']
        )

        self.val_log_keys = ['epochs'] + _prepare_log_keys(
            [self.loss_keys, self.met_keys],
            ['val']
        )
        self.log_keys += self.val_log_keys[1:]
        self.test_log_keys = ['epochs'] + _prepare_log_keys(
            [self.met_keys],
            ['test']
        )

    @exec_func_if_main_proc
    def prepare_log_file(self):
        "Prepares the log files depending on arguments"
        f = self.txt_log_file.open('a')
        cfgtxt = json.dumps(self.cfg)
        f.write(cfgtxt)
        f.write('\n\n')
        f.close()

    @exec_func_if_main_proc
    def update_log_file(self, towrite: str):
        "Updates the log files as and when required"
        with self.txt_log_file.open('a') as f:
            f.write(towrite + '\n')

    def get_predictions_list(self, predictions: Dict[str, List]) -> List[Dict]:
        "Converts dictionary of lists to list of dictionary"
        keys = list(predictions.keys())
        num_preds = len(predictions[keys[0]])
        out_list = [{k: predictions[k][ind] for k in keys}
                    for ind in range(num_preds)]
        return out_list

    def validate(self, db: Union[DataLoader, Dict[str, DataLoader]] = None,
                 mb=None, write_to_file=False) -> List[torch.tensor]:
        "Validation loop, done after every epoch"

        torch.cuda.empty_cache()
        self.mdl.eval()
        if db is None:
            dl = self.data.valid_dl
            dl_name = 'valid'
        elif isinstance(db, DataLoader):
            dl = db
            dl_name = 'valid'
        else:
            assert len(db) == 1
            dl_name = list(db.keys())[0]
            dl = db[dl_name]
        # if is_main_process():
        with torch.no_grad():
            out_loss, out_acc = self.eval_fn(
                self.mdl,
                self.loss_fn,
                dl, dl_name,
                rank=get_rank(),
                pred_path=self.predictions_dir,
                mb=mb)

        synchronize()
        if is_main_process():
            if write_to_file:
                self.logger.debug(out_loss)
                self.logger.debug(out_acc)
                out_str = '  '.join(self.val_log_keys) + '\n'
                out_list = [self.num_epoch]
                out_list += [out_loss[k] for k in self.loss_keys]
                out_list += [out_acc[k] for k in self.met_keys]
                out_str += good_format_stats(
                    self.val_log_keys, out_list
                )
                self.update_log_file(out_str)

        return out_loss, out_acc, {}

    def train_epoch(self, mb) -> List[torch.tensor]:
        "One epoch used for training"
        self.mdl.train()
        # trn_loss = SmoothenValue(0.9)
        trn_loss = SmoothenDict(self.loss_keys, 0.9)
        trn_acc = SmoothenDict(self.met_keys, 0.9)

        for batch_id, batch in enumerate(progress_bar(
                self.data.train_dl, parent=mb)):

            # Increment number of iterations
            self.num_it += 1
            for b in batch.keys():
                batch[b] = batch[b].to(self.device)
            self.optimizer.zero_grad()
            out = self.mdl(batch)
            out_loss = self.loss_fn(out, batch)
            loss = out_loss[self.loss_keys[0]]
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            # Returns original dictionary if not distributed parallel
            # loss_reduced = reduce_dict(out_loss, average=True)
            # metric_reduced = reduce_dict(metric, average=True)
            trn_loss.add_value(out_loss)

            comment_to_print = f'LossB {loss: .4f} | SmLossB {trn_loss.smooth1: .4f}'
            if self.trn_met:
                metric = self.eval_fn(out, batch)
                trn_acc.add_value(metric)
                comment_to_print += f' | AccB {trn_acc.smooth1: .4f}'
            mb.child.comment = comment_to_print
            if self.num_it % self.cfg.log.deb_it == 0:
                self.logger.debug(
                    f'Num_it {self.num_it} {trn_loss.tostring()}')
            del out_loss
            del loss
            # print(f'Done {batch_id}')
        del batch
        self.optimizer.zero_grad()
        out_loss = reduce_dict(trn_loss.smooth, average=True)
        if self.trn_met:
            out_met = reduce_dict(trn_acc.smooth, average=True)
        else:
            out_met = trn_acc.smooth
        # return trn_loss.smooth, trn_acc.smooth
        return out_loss, out_met

    def load_model_dict(self, resume_path: Optional[str] = None, load_opt: bool = False):
        "Load the model and/or optimizer"
        def check_if_mgpu_state_dict(state_dict):
            return 'module' == list(state_dict.keys())[0].split('.')[0]

        def strip_module_from_key(key):
            assert 'module' == key.split('.')[0]
            return '.'.join(key.split('.')[1:])

        if resume_path == "":
            mfile = self.model_file
        else:
            mfile = Path(resume_path)

        if not mfile.exists():
            self.logger.info(
                f'No existing model in {mfile}, starting from scratch')
            return
        try:
            checkpoint = torch.load(open(mfile, 'rb'))
            self.logger.info(f'Loaded model from {mfile} Correctly')
        except OSError as e:
            self.logger.error(
                f'Some problem with resume path: {resume_path}. Exception raised {e}')
            raise e
        if self.cfg.train['load_normally']:
            def update_k(k):
                if 'layernorm' in k and 'gamma' in k and 'frm_interact' not in k:
                    return k.replace('gamma', 'weight')
                elif 'layernorm' in k and 'beta' in k and 'frm_interact' not in k:
                    return k.replace('beta', 'bias')
                return k
            mdl_state_dict_to_load = checkpoint['model_state_dict']
            mdl_state_dict_to_load = {
                update_k(k): v for k, v in mdl_state_dict_to_load.items()}
            curr_mdl_state_dict = self.mdl.state_dict()
            checkp_mdl_mgpu = check_if_mgpu_state_dict(mdl_state_dict_to_load)
            curr_mdl_mgpu = check_if_mgpu_state_dict(curr_mdl_state_dict)

            if curr_mdl_mgpu != checkp_mdl_mgpu:
                if curr_mdl_mgpu:
                    self.mdl.module.load_state_dict(
                        mdl_state_dict_to_load,
                        strict=self.cfg.train['strict_load']
                    )
                if checkp_mdl_mgpu:
                    mdl_state_dict_to_load = {
                        strip_module_from_key(k): v
                        for k, v in mdl_state_dict_to_load.items()
                    }
                    self.mdl.load_state_dict(
                        mdl_state_dict_to_load,
                        strict=self.cfg.train['strict_load']
                    )
            else:
                self.mdl.load_state_dict(
                    mdl_state_dict_to_load,
                    strict=self.cfg.train['strict_load']
                )

        # else:
        #     load_state_dict(
        #         self.mdl, checkpoint['model_state_dict']
        #     )
        # self.logger.info('Added model file correctly')
        if 'num_it' in checkpoint.keys():
            self.num_it = checkpoint['num_it']

        if 'num_epoch' in checkpoint.keys():
            self.num_epoch = checkpoint['num_epoch']

        if 'best_met' in checkpoint.keys():
            self.best_met = checkpoint['best_met']

        if load_opt:
            self.optimizer = self.prepare_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.lr_scheduler = self.prepare_scheduler()
                self.lr_scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])

    @exec_func_if_main_proc
    def save_model_dict(self):
        "Save the model and optimizer"
        # if is_main_process():
        checkpoint = {
            'model_state_dict': self.mdl.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'num_it': self.num_it,
            'num_epoch': self.num_epoch,
            'cfgtxt': json.dumps(self.cfg),
            'best_met': self.best_met
        }
        torch.save(checkpoint, self.model_file.open('wb'))

    # @exec_func_if_main_proc
    def update_prediction_file(self, predictions, pred_file):
        rank = self.rank
        if self.cfg.do_dist:
            pred_file_to_use = pred_file.parent / f'{rank}_{pred_file.name}'
            pickle.dump(predictions, pred_file_to_use.open('wb'))
            if is_main_process() and self.cfg.do_dist:
                if pred_file.exists():
                    pred_file.unlink()
        else:
            pickle.dump(predictions, pred_file.open('wb'))
        # synchronize()
        # st_time = time.time()
        # self.rectify_predictions(pred_file)
        # end_time = time.time()
        # self.logger.info(
        #     f'Updating prediction file took time {st_time - end_time}')

    @exec_func_if_main_proc
    def rectify_predictions(self, pred_file):
        world_size = get_world_size()
        pred_files_to_use = [pred_file.parent /
                             f'{r}_{pred_file.name}' for r in range(world_size)]
        assert all([p.exists() for p in pred_files_to_use])
        out_preds = []
        for pf in pred_files_to_use:
            tmp = pickle.load(open(pf, 'rb'))
            assert isinstance(tmp, list)
            out_preds += tmp
        pickle.dump(out_preds, pred_file.open('wb'))

    def prepare_to_write(
            self,
            train_loss: Dict[str, torch.tensor],
            train_acc: Dict[str, torch.tensor],
            val_loss: Dict[str, torch.tensor] = None,
            val_acc: Dict[str, torch.tensor] = None,
            key_list: List[str] = None
    ) -> List[torch.tensor]:
        if key_list is None:
            key_list = self.log_keys

        epoch = self.num_epoch
        out_list = [epoch]

        out_list += [train_loss[k] for k in self.loss_keys]
        if val_loss is not None:
            out_list += [val_loss[k] for k in self.loss_keys]

        # for k in self.met_keys:
        if train_acc is not None:
            out_list += [train_acc[k] for k in self.met_keys]
        if val_acc is not None:
            out_list += [val_acc[k] for k in self.met_keys]

        assert len(out_list) == len(key_list)
        return out_list

    @property
    def lr(self):
        return self.cfg.train['lr']

    @property
    def epoch(self):
        return self.cfg.train['epochs']

    @exec_func_if_main_proc
    def master_bar_write(self, mb, **kwargs):
        mb.write(**kwargs)

    def fit(self, epochs: int, lr: float,
            params_opt_dict: Optional[Dict] = None):
        self.update_log_file('  '.join(self.log_keys) + '\n')
        "Main training loop"
        # Print logger at the start of the training loop
        self.logger.info(self.cfg)
        # Initialize the progress_bar
        mb = master_bar(range(epochs))
        # Initialize optimizer
        # Prepare Optimizer may need to be re-written as per use
        self.optimizer = self.prepare_optimizer(params_opt_dict)
        # Initialize scheduler
        # Prepare scheduler may need to re-written as per use
        self.lr_scheduler = self.prepare_scheduler(self.optimizer)

        # Write the top row display
        # mb.write(self.log_keys, table=True)
        self.master_bar_write(mb, line=self.log_keys, table=True)
        exception = False
        met_to_use = None
        # Keep record of time until exit
        st_time = time.time()
        try:
            # Loop over epochs
            for epoch in mb:
                self.num_epoch += 1
                train_loss, train_acc = self.train_epoch(mb)
                synchronize()
                valid_loss, valid_acc, _ = self.validate(
                    self.data.valid_dl, mb)
                synchronize()
                valid_acc_to_use = valid_acc[self.met_keys[0]]
                # Depending on type
                self.scheduler_step(valid_acc_to_use)

                # Now only need main process
                # Decide to save or not
                met_to_use = valid_acc[self.met_keys[0]].cpu()
                if self.best_met < met_to_use:
                    self.best_met = met_to_use
                    self.save_model_dict()

                synchronize()
                # Prepare what all to write
                to_write = self.prepare_to_write(
                    train_loss, None,
                    valid_loss, valid_acc
                )
                synchronize()
                # Display on terminal
                assert to_write is not None
                mb_write = [str(stat) if isinstance(stat, int)
                            else f'{stat:.4f}' for stat in to_write]
                self.master_bar_write(mb, line=mb_write, table=True)

                # for k, record in zip(self.log_keys, to_write):
                #     self.writer.add_scalar(
                #         tag=k, scalar_value=record, global_step=self.num_epoch)
                # Update in the log file
                self.update_log_file(
                    good_format_stats(self.log_keys, to_write))
                synchronize()
        except Exception as e:
            exception = e
            raise e
        finally:
            end_time = time.time()
            self.update_log_file(
                f'epochs done {epoch}. Exited due to exception {exception}. '
                f'Total time taken {end_time - st_time: 0.4f}\n\n'
            )
            # Decide to save finally or not
            if met_to_use:
                if self.best_met < met_to_use:
                    self.save_model_dict()
            synchronize()

    def testing(self, db: Dict[str, DataLoader]):
        if isinstance(db, DataLoader):
            db = {'dl0': db}
        for dl_name, dl in tqdm(db.items(), total=len(db)):
            out_loss, out_acc, preds = self.validate(dl)

            log_keys = self.val_log_keys

            to_write = self.prepare_to_write(
                out_loss, out_acc, key_list=log_keys)
            header = '  '.join(log_keys) + '\n'
            self.update_log_file(header)
            self.update_log_file(good_format_stats(
                log_keys, to_write))

            self.logger.info(header)
            self.logger.info(good_format_stats(log_keys, to_write))

            self.update_prediction_file(
                preds, self.predictions_dir / f'{dl_name}_preds.pkl')

    def prepare_optimizer(self, params=None):
        "Prepare a normal optimizer"
        if not params:
            params = self.mdl.parameters()
        opt = self.opt_fn(params, lr=self.lr)
        return opt

    def prepare_scheduler(self, opt: torch.optim):
        "Prepares a LR scheduler on top of optimizer"
        self.sched_using_val_metric = self.cfg.train.use_reduce_lr_plateau
        if self.sched_using_val_metric:
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=self.cfg.reduce_factor, patience=self.cfg.patience)
        else:
            lr_sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda epoch: 1)

        return lr_sched

    def scheduler_step(self, val_metric):
        if self.sched_using_val_metric:
            self.lr_scheduler.step(val_metric)
        else:
            self.lr_scheduler.step()
        return

    def overfit_batch(self, epochs: int, lr: float):
        "Sanity check to see if model overfits on a batch"
        guess = True
        idx = 0
        # diter = iter(self.data.valid_dl)
        diter = iter(self.data.train_dl)
        if self.cfg.ds.ds4_type != 'single':
            while guess:
                batch = next(diter)
                print(idx, 'why')
                idx += 1
                if torch.all(batch['num_cmp'] != 1):
                    guess = False
        else:
            batch = next(diter)

        for b in batch.keys():
            batch[b] = batch[b].to(self.device)
        self.mdl.train()
        opt = self.prepare_optimizer()

        for i in range(epochs):
            opt.zero_grad()
            out = self.mdl(batch)
            out_loss = self.loss_fn(out, batch)
            loss = out_loss[self.loss_keys[0]]
            loss = loss.mean()
            loss.backward()
            opt.step()
            # met = self.eval_f n(out, batch)
            out_str = f'Iter {i} | loss {loss: 0.4f}'
            out_str += ' | '.join([f'{k}: {v.mean()}' for k,
                                   v in out_loss.items()])
            self.logger.debug(out_str)
            print(out_str)
            # print(f'Iter {i} | loss {loss: 0.4f} | acc {met: 0.4f}')
