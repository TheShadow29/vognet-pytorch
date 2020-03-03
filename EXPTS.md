# Reproducing Numbers in the Paper:

Make sure you are the root of the project.

## Main Results (GT5/P100)

You need `ds.exp_setting='gt5'` or `ds.exp_setting='p100'` in `anet_srl_cfg.yml`

Note that the string after `code/main_dist.py` is arbitrary, and you can set it to anything
you want.

1. `python code/main_dist.py "svsq_igrnd" --ds.conc_type='svsq' --mdl.name='igrnd --train.prob_thresh=0.'`
1. `python code/main_dist.py "svsq_vgrnd" --ds.conc_type='svsq' --mdl.name='vgrnd --train.prob_thresh=0.'`
1. `python code/main_dist.py "svsq_vog" --ds.conc_type='svsq' --mdl.name='vog --train.prob_thresh=0.`

1. `python code/main_dist.py "sep_igrnd" --ds.conc_type='sep' --mdl.name='igrnd' --train.prob_thresh=0.`
1. `python code/main_dist.py "sep_vgrnd" --ds.conc_type='sep' --mdl.name='vgrnd' --train.prob_thresh=0.`
1. `python code/main_dist.py "sep_vog" --ds.conc_type='sep' --mdl.name='vog' --train.prob_thresh=0.`

1. `python code/main_dist.py "temp_igrnd" --ds.conc_type='temp' --mdl.name='igrnd' --train.prob_thresh=0.2`
1. `python code/main_dist.py "temp_vgrnd" --ds.conc_type='temp' --mdl.name='vgrnd' --train.prob_thresh=0.2`
1. `python code/main_dist.py "temp_vog" --ds.conc_type='temp' --mdl.name='vog' --train.prob_thresh=0.2`

1. `python code/main_dist.py "spat_igrnd" --ds.conc_type='spat' --mdl.name='igrnd' --train.prob_thresh=0.2`
1. `python code/main_dist.py "spat_vgrnd" --ds.conc_type='spat' --mdl.name='vgrnd' --train.prob_thresh=0.2`
1. `python code/main_dist.py "spat_vog" --ds.conc_type='spat' --mdl.name='vog' --train.prob_thresh=0.2`

## GT5 -> P100 Transfer

For transfering GT5 trained models to P100, we need to pass `train.resume=True` and
provide the resume path via `train.resume_path`.
We also need to provide `only_val=True` and set `ds.exp_setting='p100'`

For instance, to test a ImageGrnd in `p100` setting which was trained in `gt5` setting:

1. `python code/main_dist.py "svsq_igrnd_gt5_to_p100" --ds.conc_type='svsq' --mdl.name='igrnd --train.prob_thresh=0.' --train.resume=True --train.resume_path='./tmp/models/svsq_igrnd.pth' --ds.exp_setting='p100' --only_val=True`

## Across Conc Types

You can also test models trained in one concatenation type like `SPAT` in another type like `TEMP`. For instance,

1. `python code/main_dist.py "vog_train_spat_test_temp" --ds.conc_type='temp' --mdl.name='vog' --train.prob_thresh=0.2 --train.resume=True --train.resume_path='./tmp/models/spat_vog.pth --only_val=True`

## Model Ablations

If you want to use no object or multi-modal transformer, use `--mdl.name='igrnd'`
If you want only object transformer, use `--mdl.name='vgrnd'`
If you want only multimodal transformer, use `--mdl.name='vog'` and `--mdl.obj_tx.to_use=false`
If you want to use both object and multimodal transformer use `--mdl.name='vog'`

To set the number of heads and layers, set `n_layers` and `n_heads` under `obj_tx` and `mul_tx` respectively.

To use relative position encoding set `use_rel=true` under `obj_tx` and/or `mul_tx`.
