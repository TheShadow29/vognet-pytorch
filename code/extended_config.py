from yacs.config import CfgNode as CN
# import json
# import yaml
from _init_stuff import yaml
from typing import Dict, Any

# ds_info = CN(json.load(open('./configs/ds_info.json')))
# def_cfg = CN(json.load(open('./configs/cfg.json')))
with open('./configs/anet_ent_cfg.yaml') as f:
    def_cfg = yaml.safe_load(f)

cfg = CN(def_cfg)
cfg.comm = CN()


# Device
# setting default device
cfg.device = 'cuda'

# Training
cfg.local_rank = 0
cfg.do_dist = False

# Testing
cfg.only_val = False
cfg.only_test = False

# Mdl
cfg.pretrained_path = './save/anet-sup-0.05-0-0.1-run1/model-best.pth'

key_maps = {}


def create_from_dict(dct: Dict[str, Any], prefix: str, cfg: CN):
    """
    Helper function to create yacs config from dictionary
    """
    dct_cfg = CN(dct, new_allowed=True)
    prefix_list = prefix.split('.')
    d = cfg
    for pref in prefix_list[:-1]:
        assert isinstance(d, CN)
        if pref not in d:
            setattr(d, pref, CN())
        d = d[pref]
    if hasattr(d, prefix_list[-1]):
        old_dct_cfg = d[prefix_list[-1]]
        dct_cfg.merge_from_other_cfg(old_dct_cfg)

    setattr(d, prefix_list[-1], dct_cfg)
    return cfg


def update_from_dict(cfg: CN, dct: Dict[str, Any],
                     key_maps: Dict[str, str] = None) -> CN:
    """
    Given original CfgNode (cfg) and input dictionary allows changing
    the cfg with the updated dictionary values
    Optional key_maps argument which defines a mapping between
    same keys of the cfg node. Only used for convenience
    Adapted from:
    https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L219
    """
    # Original cfg
    root = cfg
    if key_maps is None:
        key_maps = []
    # Change the input dictionary using keymaps
    # Now it is aligned with the cfg
    full_key_list = list(dct.keys())
    for full_key in full_key_list:
        if full_key in key_maps:
            cfg[full_key] = dct[full_key]
            new_key = key_maps[full_key]
            dct[new_key] = dct.pop(full_key)

    # Convert the cfg using dictionary input
    for full_key, v in dct.items():
        if root.key_is_deprecated(full_key):
            continue
        if root.key_is_renamed(full_key):
            root.raise_key_rename_error(full_key)
        key_list = full_key.split(".")
        d = cfg
        for subkey in key_list[:-1]:
            # Most important statement
            assert subkey in d, f'key {full_key} doesnot exist'
            d = d[subkey]

        subkey = key_list[-1]
        # Most important statement
        assert subkey in d, f'key {full_key} doesnot exist'

        value = cfg._decode_cfg_value(v)

        assert isinstance(value, type(d[subkey]))
        d[subkey] = value

    return cfg


def post_proc_config(cfg: CN):
    if cfg.ds.add_prop_to_region:
        cfg.mdl.att_feat_size += 5
    if cfg.ds.use_gt_prop:
        if cfg.ds.ngt_prop == 10:
            cfg.ds.proposal_h5 = cfg.ds.proposal_gt10_h5
            cfg.ds.feature_root = cfg.ds.feature_gt10_root
            cfg.misc.num_prop_per_frm = 10
        elif cfg.ds.ngt_prop == 5:
            cfg.ds.proposal_h5 = cfg.ds.proposal_gt5_h5
            cfg.ds.feature_root = cfg.ds.feature_gt5_root
            cfg.misc.num_prop_per_frm = 5
        else:
            raise NotImplementedError
    return cfg
    pass

# def get_config_after_kwargs(cfg, kwargs: Dict[str, Any]):
# ds_info = CN(json.load(open('./configs/ds_info.json')))
# def_cfg = CN(json.load(open('./configs/cfg.json')))

#     upd_cfg = update_from_dict(def_cfg, kwargs, key_maps)

#     cfg_dict = {
#         'ds_to_use': upd_cfg.ds_to_use,
#         'mdl_to_use': upd_cfg.mdl_to_use,
#         'lfn_to_use': upd_cfg.lfn_to_use,
#         'efn_to_use': upd_cfg.efn_to_use,
#         'opt_to_use': upd_cfg.opt_to_use,
#         'sfn_to_use': upd_cfg.sfn_to_use
#     }

#     cfg = CN(cfg_dict)

#     cfg = create_from_dict(ds_info[cfg.ds_to_use], 'DS', cfg)
