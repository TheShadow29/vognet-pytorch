"""
Main file for distributed training
"""
import sys
# from dat_loader import get_data
from dat_loader_simple import get_data
from mdl_selector import get_mdl_loss_eval
from trn_utils import Learner, synchronize

import torch
import fire
from functools import partial

from extended_config import (
    cfg as conf,
    key_maps,
    CN,
    update_from_dict,
    post_proc_config
)

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def get_name_from_inst(inst):
    return inst.__class__.__name__


def learner_init(uid: str, cfg: CN) -> Learner:
    # = get_mdl_loss(cfg)
    mdl_loss_eval = get_mdl_loss_eval(cfg)
    get_default_net = mdl_loss_eval['mdl']
    get_default_loss = mdl_loss_eval['loss']
    get_default_eval = mdl_loss_eval['eval']

    device = torch.device('cuda')
    # device = torch.device('cpu')
    data = get_data(cfg)
    comm = data.train_dl.dataset.comm
    mdl = get_default_net(cfg=cfg, comm=comm)

    # pretrained_state_dict = torch.load(cfg.pretrained_path)
    # to_load_state_dict = pretrained_state_dict
    # mdl.load_state_dict(to_load_state_dict)

    loss_fn = get_default_loss(cfg, comm)
    loss_fn.to(device)
    # if cfg.do_dist:
    # loss_fn.to(device)

    eval_fn = get_default_eval(cfg, comm, device)
    eval_fn.to(device)
    opt_fn = partial(torch.optim.Adam, betas=(0.9, 0.99))

    # unfreeze cfg to save the names
    cfg.defrost()
    module_name = mdl
    cfg.mdl_data_names = CN({
        'trn_data': get_name_from_inst(data.train_dl.dataset),
        'val_data': get_name_from_inst(data.valid_dl.dataset),
        'trn_collator': get_name_from_inst(data.train_dl.collate_fn),
        'val_collator': get_name_from_inst(data.valid_dl.collate_fn),
        'mdl_name': get_name_from_inst(module_name),
        'loss_name': get_name_from_inst(loss_fn),
        'eval_name': get_name_from_inst(eval_fn),
        'opt_name': opt_fn.func.__name__
    })
    cfg.freeze()

    learn = Learner(uid=uid, data=data, mdl=mdl, loss_fn=loss_fn,
                    opt_fn=opt_fn, eval_fn=eval_fn, device=device, cfg=cfg)

    if cfg.do_dist:
        mdl.to(device)
        mdl = torch.nn.parallel.DistributedDataParallel(
            mdl, device_ids=[cfg.local_rank],
            output_device=cfg.local_rank, broadcast_buffers=True,
            find_unused_parameters=True)
    elif cfg.do_dp:
        # Use data parallel
        mdl = torch.nn.DataParallel(mdl)

    mdl = mdl.to(device)

    return learn


def main_dist(uid: str, **kwargs):
    """
    uid is a unique identifier for the experiment name
    Can be kept same as a previous run, by default will start executing
    from latest saved model
    **kwargs: allows arbit arguments of cfg to be changed
    """
    cfg = conf
    num_gpus = torch.cuda.device_count()
    cfg.num_gpus = num_gpus
    cfg.uid = uid
    cfg.cmd = sys.argv
    if num_gpus > 1:
        if 'local_rank' in kwargs:
            # We are doing distributed parallel
            cfg.do_dist = True
            torch.cuda.set_device(kwargs['local_rank'])
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()
        else:
            # We are doing data parallel
            cfg.do_dist = False
            # cfg.do_dp = True
    # Update the config file depending on the command line args
    cfg = update_from_dict(cfg, kwargs, key_maps)
    cfg = post_proc_config(cfg)
    # Freeze the cfg, can no longer be changed
    cfg.freeze()
    # print(cfg)
    # Initialize learner
    learn = learner_init(uid, cfg)
    # Train or Test
    if not (cfg.only_val or cfg.only_test or cfg.overfit_batch):
        learn.fit(epochs=cfg.train.epochs, lr=cfg.train.lr)
    else:
        if cfg.overfit_batch:
            learn.overfit_batch(1000, 1e-4)
        if cfg.only_val:
            val_loss, val_acc, _ = learn.validate(
                db={'valid': learn.data.valid_dl},
                write_to_file=True
            )
            print(val_loss)
            print(val_acc)
            # learn.testing(learn.data.valid_dl)
            pass
        if cfg.only_test:
            # learn.testing(learn.data.test_dl)
            test_loss, test_acc, _ = learn.validate(
                db=learn.data.test_dl)
            print(test_loss)
            print(test_acc)

    return learn


if __name__ == '__main__':
    learn = fire.Fire(main_dist)
