# Reproducing Numbers in the Paper:

Make sure you are at the root of the project.

Basic Usage is
```
python code/main_dist.py $exp_name --arg1=val1 --arg2.subarg1=val2 --arg3.subarg2.subsubarg1=val3
```
Here the arguments follow the hierarchy defined in the config file.


In most cases, need to set four things: 
1. `ds.exp_setting` as `gt5` (using 5 proposals per frame including ground-truth) or `p100` (using 100 proposals per frame)
1. `mdl.name` as `igrnd` (ImgGrnd), `vgrnd` (VidGrnd), `vog` (VOGNet). For `vog` model, you need to explicitly add 
`--mdl.obj_tx.use_rel=True` and  `--mdl.mul_tx.use_rel=True` as they are set to `False` by default in the config file.
1. `ds.conc_type` as `svsq`, `sep`, `temp` or `spat` depending on the concatenation strategy to use.
1. `train.prob_thresh` which is the probability threshold used for evaluation. Note that `svsq`, `sep` use a threshold=0.

All default hyper-parameters defined in `configs/anet_srl_cfg.yml` can be similarly changed in command-line itself.

Note that the string after `code/main_dist.py` is arbitrary, and you can set it to anything
you want.

## Main Results (GT5/P100)

To run ImgGrnd with `SVSQ` strategy:
```
python code/main_dist.py "svsq_igrnd" --ds.conc_type='svsq' --mdl.name='igrnd' --train.prob_thresh=0.
```

Similarly, to run ImgGrnd with `SEP`, `TEMP`, `SPAT`, set `--ds.conc_type` 

```python code/main_dist.py "sep_igrnd" --ds.conc_type='sep' --mdl.name='igrnd' --train.prob_thresh=0.```

```python code/main_dist.py "temp_igrnd" --ds.conc_type='temp' --mdl.name='igrnd' --train.prob_thresh=0.2```

```python code/main_dist.py "spat_igrnd" --ds.conc_type='spat' --mdl.name='igrnd' --train.prob_thresh=0.2```

Similary, to run VidGrnd in `SPAT`
```
python code/main_dist.py "spat_vgrnd" --ds.conc_type='spat' --mdl.name='vgrnd' --train.prob_thresh=0.2
```

Or, to run VOGNet in `SPAT`
```
python code/main_dist.py "spat_vog" --ds.conc_type='spat' --mdl.name='vog' \
--mdl.obj_tx.use_rel=True --mdl.mul_tx.use_rel=True --train.prob_thresh=0.2
```

To run with 100 proposals per frame, additionally pass `--ds.exp_setting='p100'` 
For `TEMP` and `SPAT` in `p100` we set `prob_thresh=0.1` (by tuning on validation set).

## Across Conc Types

You can also test models trained in one concatenation type like `SPAT` in another type like `TEMP`. For instance,

```
python code/main_dist.py "vog_train_spat_test_temp" --ds.conc_type='temp' --mdl.name='vog' --train.prob_thresh=0.2 --train.resume=True --train.resume_path='./tmp/models/spat_vog.pth --only_val=True
```

## Model Ablations

1. No object or multi-modal transformer, use `--mdl.name='igrnd'`
1. Only object transformer, use `--mdl.name='vgrnd'`
1. Only multimodal transformer, use `--mdl.name='vog'` and `--mdl.obj_tx.to_use=false`
1. Both object and multimodal transformer with RPE use `--mdl.name='vog'`

To set the number of heads and layers, set `n_layers` and `n_heads` under `obj_tx` and `mul_tx` respectively.

To use relative position encoding set `use_rel=true` under `obj_tx` and/or `mul_tx`.

## GT5 -> P100 Transfer

For transfering GT5 trained models to P100, we need to pass `train.resume=True` and
provide the trained moel path via `train.resume_path`.

We also need to provide `only_val=True` and set `ds.exp_setting='p100'`

For instance, to test a ImageGrnd in `p100` setting which was trained in `gt5` setting:

```
python code/main_dist.py "svsq_igrnd_gt5_to_p100" --ds.conc_type='svsq' --mdl.name='igrnd --train.prob_thresh=0.' --train.resume=True --train.resume_path='./tmp/models/svsq_igrnd.pth' --ds.exp_setting='p100' --only_val=True
```
Here, `./tmp/models/svsq_igrnd.pth` is the model path for ImgGrnd trained using SVSQ in GT5 setting.

For `TEMP` and `SPAT` we found `train.prob_thresh=0.5` to give the best results

# Pre-Trained Models

We provide google drive links to the best model, the output predictions for all the tables in the paper. 
Alternatively, you can download them at once from [this drive link](https://drive.google.com/open?id=1e3FiX4FTC8n6UrzY9fTYQzFNKWHihzoQ)

Additionally, the exact command used to run the model can be found in the log file under `cmd`

## GT5 Models, All Conc Strategies
Page 7, Table 3, Row 1

| Conc Strategy   | Model Type   | ID                    | []()                                                                        | []()                                                                      | []()                                                                       |
|-----------------|--------------|-----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| SVSQ            | ImgGrnd      | svsq_igrnd_gt5_2Mar20 | [model](https://drive.google.com/open?id=1qwVEFB4H5XH8w3LjsKO1MnSVSSWR4Bvp) | [log](https://drive.google.com/open?id=11MlEXegyiJcZeFrI3VCfYwj0GRJMCaHu) | [pred](https://drive.google.com/open?id=1TOdfYsJDy9gMqELwqX-s32bocxfN8Rn5) |
| SVSQ            | VidGrnd      | svsq_vgrnd_3Mar20     | [model](https://drive.google.com/open?id=14t1ehKVI4Ek71LsgeMkWJ4KZzc29BSUp) | [log](https://drive.google.com/open?id=1ZFv0aL--CGAL_I_7vQLJBh3tZ8ZmGARG) | [pred](https://drive.google.com/open?id=1SNGH-L1s-lu5dQ5Wget4IS-_EmPsZXgw) |
| SVSQ            | VOGNet       | svsq_vog_3Mar20       | [model](https://drive.google.com/open?id=13yoYr4VoV46Xi8ZXqC0txYgKNIpwExMX) | [log](https://drive.google.com/open?id=1WQ_57aKaBkrCstMFrAxkx5pSSq_Q3dG-) | [pred](https://drive.google.com/open?id=1eqFs4j6eXwbp3-1GjWejHh_5Yb9i1llX) |
| SEP             | ImgGrnd      | sep_igrnd_gt5_2Mar20  | [model](https://drive.google.com/open?id=1L0CncIMS3CMu2zmVRZi22cGD1j5yK3Ng) | [log](https://drive.google.com/open?id=1iq1iC3D2MuhOZWuf98riQs9n6mMxngZ4) | [pred](https://drive.google.com/open?id=1hLZBGHxZxZXL4UTdiK7-VFtyzU21JKvd) |
| SEP             | VidGrnd      | sep_vgrnd_3Mar20      | [model](https://drive.google.com/open?id=1d4IR7rssKr3JATkkh9_u8UXoF9GKoKaL) | [log](https://drive.google.com/open?id=1OVcJD8WsTrrWYs_HuimFSy2qwf-lUsLy) | [pred](https://drive.google.com/open?id=1U6Ls4r9nV30qYOAp7TROpcGHrUqXgtv5) |
| SEP             | VOGNet       | sep_vog_3Mar20        | [model](https://drive.google.com/open?id=151w2JChasME2zo_2u9Be1LQsJ3AN5LvQ) | [log](https://drive.google.com/open?id=1CTjmSW3XU36gPu3roiiuZsWjIgsfVcW5) | [pred](https://drive.google.com/open?id=1syML07HaMYEjFTOZkuy7LqO7l_raiYHK) |
| TEMP            | ImgGrnd      | temp_igrnd_2Mar20     | [model](https://drive.google.com/open?id=1Kw3zMbkXUm3l0sUGN5XvxKwofRKh3vD2) | [log](https://drive.google.com/open?id=10RRtZ80KAFo8kOJodCrDztAqbaaF4yog) | [pred](https://drive.google.com/open?id=1hnD-ZEP9kd3qOcAsbHbyEbFc6Ndvxnhs) |
| TEMP            | VidGrnd      | temp_vgrnd_3Mar20     | [model](https://drive.google.com/open?id=1Oauzi5E2dINnN2VWajPh7uBLfT20_wbw) | [log](https://drive.google.com/open?id=18OQstok3TZwjkmda5OT_CjcO6x-cxZJK) | [pred](https://drive.google.com/open?id=1Uqb9dtFpNGbWi0z1btpzfavbSaM235Vi) |
| TEMP            | VOGNet       | temp_vog_3Mar20       | [model](https://drive.google.com/open?id=1KGq0x4X3reRguvrtnxJ4YvMcOi2lDpsv) | [log](https://drive.google.com/open?id=12RLUr4037k33fzQjEfd6m4IU52IFizv_) | [pred](https://drive.google.com/open?id=1zT8XZxNGMD9L0GuSM--XLGAJDFsIwhjO) |
| SPAT            | ImgGrnd      | spat_igrnd_3Mar20     | [model](https://drive.google.com/open?id=1uH_LLQRegugRA_z7QILDuauEauk7M_QG) | [log](https://drive.google.com/open?id=1KK3750SuC8Ra0qvLmpODuS-EOnAE9Coy) | [pred](https://drive.google.com/open?id=1Pn_MuBKHb5kvShHn2YU1mHmuLtyW2OY5) |
| SPAT            | VidGrnd      | spat_vgrnd_3Mar20     | [model](https://drive.google.com/open?id=1zLfVG8Uu4n9s1_iFglVrez4-Rd_4Zy9I) | [log](https://drive.google.com/open?id=1nxWJb3vSMwe8xD4HmZRIHrKXbHtCanwl) | [pred](https://drive.google.com/open?id=1745DOGTbvvHg34Xkf6NKw_c3c5yOQAM_) |
| SPAT            | VOGNet       | spat_vog_3Mar20       | [model](https://drive.google.com/open?id=1jV3GhJLCDcHGDkj2NHUaWCz5RClwxoN7) | [log](https://drive.google.com/open?id=1_TRlAkqaYL0zg0zYnMmDMf_QkawOvZPr) | [pred](https://drive.google.com/open?id=1OaXFKF7ZhR8ziFbL088Vtct6lb8ZXSI2) |


## P100 Models, All Conc Strategies
Page 7, Table 3, Row 2

| Conc Strategy   | Model Type   | ID                      | []()                                                                        | []()                                                                      | []()                                                                       |
|-----------------|--------------|-------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| SVSQ            | ImgGrnd      | svsq_igrnd_p100_10Mar20 | [model](https://drive.google.com/open?id=1MRGiNSZ5ennELgj-dqjGkkm8fjPl01ZT) | [log](https://drive.google.com/open?id=11KFW9nMUZxv4x4535_oRQvfRTSQRlewA) | [pred](https://drive.google.com/open?id=1-3i1duL2-3g-edui1VDmKlZaZgq1uann) |
| SVSQ            | VidGrnd      | svsq_vgrnd_p100_10Mar20 | [model](https://drive.google.com/open?id=1_cn4OWMb_arBfKlOpOzSnfHhuNo0f3Gx) | [log](https://drive.google.com/open?id=1pFlAIzwAgWCU6bZYrtAbmeSyuxOnNC3V) | [pred](https://drive.google.com/open?id=1UQCpolH_n1xzOWpPdMeMzFqwkPtkckhk) |
| SVSQ            | VOGNet       | svsq_vog_p100_10Mar20   | [model](https://drive.google.com/open?id=1rUJSZav4KB624VjI_E6F8sePZglEw85Z) | [log](https://drive.google.com/open?id=1OQWnUSXnZX-QC_dw-4glkmabtEj2rvLi) | [pred](https://drive.google.com/open?id=1qlm3ePo19p6sCDn7n_BtUc3MskD0SCO5) |
| SEP             | ImgGrnd      | sep_igrnd_p100_11Mar20  | [model](https://drive.google.com/open?id=1r0e2jNXEX5YZ6yE2VLVTNYkktYCaVcSK) | [log](https://drive.google.com/open?id=1tOSAor5LAyvXJ6QA0OTFtOwIHm6yBmaJ) | [pred](https://drive.google.com/open?id=18Mwj2fhH2Q0gE5sQEZGAjmZcZcIbtm4j) |
| SEP             | VidGrnd      | sep_vgrnd_p100_8Mar20   | [model](https://drive.google.com/open?id=1ILigaYdfyoZ9gpAZr4GOB40nNvFXbR3O) | [log](https://drive.google.com/open?id=1OX3nJeVvWkWO-yCOVzbSoEusY_q_djA_) | [pred](https://drive.google.com/open?id=1u4B62UoonyQpTTdv0niDBpjddIiKJkOb) |
| SEP             | VOGNet       | sep_vog_p100_6Mar20     | [model](https://drive.google.com/open?id=1V2hnY4mOVN5h9OHacz0gt8KlWaJ1Na7B) | [log](https://drive.google.com/open?id=1BTMgsRhgqiwFkbRB-WrpmfcGYSWFaWv2) | [pred](https://drive.google.com/open?id=10RLaf09-GS0mnFBbT6uIKVDLJyFz4Aym) |
| TEMP            | ImgGrnd      | temp_igrnd_p100_11Mar20 | [model](https://drive.google.com/open?id=1dN7xSh8Wq4bqTNYcYDXhVUzU3xC7YvUJ) | [log](https://drive.google.com/open?id=1T7subHy8ShIVLQR9YhO4IhqXzGOA1luF) | [pred](https://drive.google.com/open?id=1pEcdp3pfGbabQZLkSklfm_2Amb0MbIEC) |
| TEMP            | VidGrnd      | temp_vgrnd_p100_8Mar20  | [model](https://drive.google.com/open?id=10kk3LFVYMXzZNmx7IKRr-O4IF7SssJ5N) | [log](https://drive.google.com/open?id=1JGTWieT1y1wI4J_XZYZxNBACrmtZqo_W) | [pred](https://drive.google.com/open?id=1PSaRxkJz8xIZNCfzQ0unHcnCPS6JtPDW) |
| TEMP            | VOGNet       | temp_vog_p100_6Mar20    | [model](https://drive.google.com/open?id=1sNJWidQmjtYHLfYG1iVK6x0aS2eZzItx) | [log](https://drive.google.com/open?id=1AbecuzsonymYoac0zNw745VO-HUwPOk4) | [pred](https://drive.google.com/open?id=1hNDEi2nX-P1Fge72TY7RGWfnsbnKU3uC) |
| SPAT            | ImgGrnd      | spat_igrnd_p100_11Mar20 | [model](https://drive.google.com/open?id=1SK8p9E8gSWhzTrBP9J_cv6w2WS8rSb75) | [log](https://drive.google.com/open?id=1HsuJ9ZmoteyGDC_zc8_PeoAq9DEkpwcF) | [pred](https://drive.google.com/open?id=1lZ7kdhlsulEiThTwsioOdy-qnSVMReLn) |
| SPAT            | VidGrnd      | spat_vgrnd_p100_8Mar20  | [model](https://drive.google.com/open?id=1cH4l0DlzYkyd8A5o7M0nJ7j63tn_drDI) | [log](https://drive.google.com/open?id=1ncYJrx8hWD1Bmns1EjfCeo9ln7oPJVve) | [pred](https://drive.google.com/open?id=12hG4RoXUNWESiXRk9vDFKzddbu8EFxnk) |
| SPAT            | VOGNet       | spat_vog_p100_6Mar20    | [model](https://drive.google.com/open?id=1OX4nNSwTzeiDdsYWfOaJp6g0TvVxbywc) | [log](https://drive.google.com/open?id=1bFKpObdYkgzhruYj--Fj-H7N2OjII-3X) | [pred](https://drive.google.com/open?id=1rShlnxiPhCzkmRfEqTJAG0xqZPhymBk8) |


## GT5 Models, Cross Performance
Page 7, Table 4

| Train Conc Strategy   | Test Conc Strategy   | ID                                 | []()                                                                        | []()                                                                      | []()                                                                       |
|-----------------------|----------------------|------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| SVSQ                  | SVSQ                 | svsq_vog_3Mar20                    | [model](https://drive.google.com/open?id=13yoYr4VoV46Xi8ZXqC0txYgKNIpwExMX) | [log](https://drive.google.com/open?id=1WQ_57aKaBkrCstMFrAxkx5pSSq_Q3dG-) | [pred](https://drive.google.com/open?id=1eqFs4j6eXwbp3-1GjWejHh_5Yb9i1llX) |
| SVSQ                  | TEMP                 | vog_gt5_train_svsq_val_temp_4Mar20 | -                                                                           | [log](https://drive.google.com/open?id=1qV_w6Z7K-AkoYfzn9nwWovqhXo7e4DeI) | [pred](https://drive.google.com/open?id=1FsV7UsQBuezT5WJV1q2yGjU9llmGewvc) |
| SVSQ                  | SPAT                 | vog_gt5_train_svsq_val_spat_4Mar20 | -                                                                           | [log](https://drive.google.com/open?id=1DttlYKgzfL_mp14kcddw_-FCnncr0uqI) | [pred](https://drive.google.com/open?id=1XdCnjZZwCHPhN-dxTSo_I3CwMt7znLew) |
| TEMP                  | SVSQ                 | vog_gt5_train_temp_val_svsq_4Mar20 | -                                                                           | [log](https://drive.google.com/open?id=1veVXcL9I32cERklXlbrw15Cd6S8AoV40) | [pred](https://drive.google.com/open?id=1NgoXBR16PbeeiqzrQSF0d7IMsh4lIROa) |
| TEMP                  | SPAT                 | vog_gt5_train_temp_val_spat_4Mar20 | -                                                                           | [log](https://drive.google.com/open?id=17P6xU4MV9CPBA-JKFMw_C9F1F2uOgcQ1) | [pred](https://drive.google.com/open?id=1TFYAtkNKJ_XpGCunS76EFugCrEUTNH06) |
| TEMP                  | TEMP                 | temp_vog_3Mar20                    | [model](https://drive.google.com/open?id=1KGq0x4X3reRguvrtnxJ4YvMcOi2lDpsv) | [log](https://drive.google.com/open?id=12RLUr4037k33fzQjEfd6m4IU52IFizv_) | [pred](https://drive.google.com/open?id=1zT8XZxNGMD9L0GuSM--XLGAJDFsIwhjO) |
| SPAT                  | SVSQ                 | vog_gt5_train_spat_val_svsq_4Mar20 | -                                                                           | [log](https://drive.google.com/open?id=1jtq1FSuQ7OJiKUVEipRM0LtAv1Q2jSTQ) | [pred](https://drive.google.com/open?id=1-wLaPGq2GeKD50zR769dTKiV3bmeRMEt) |
| SPAT                  | TEMP                 | vog_gt5_train_spat_val_temp_4Mar20 | -                                                                           | [log](https://drive.google.com/open?id=1gIGzj2QquGWltzyeuf6sh4yrDgvfOMBs) | [pred](https://drive.google.com/open?id=1vyrG-Difiz9T1N_pkPFDAGYrodK_VeKU) |
| SPAT                  | SPAT                 | spat_vog_3Mar20                    | [model](https://drive.google.com/open?id=1jV3GhJLCDcHGDkj2NHUaWCz5RClwxoN7) | [log](https://drive.google.com/open?id=1_TRlAkqaYL0zg0zYnMmDMf_QkawOvZPr) | [pred](https://drive.google.com/open?id=1OaXFKF7ZhR8ziFbL088Vtct6lb8ZXSI2) |


## GT5 Models, Sampling 
Page 7, Table 5

| Train Sampling   | Test Sampling   | Strategy   | ID                                     | []()                                                                        | []()                                                                      | []()                                                                       |
|------------------|-----------------|------------|----------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Rnd              | CS              | SEP        | sep_vog_gt5_rand_samp_4Mar20           | [model](https://drive.google.com/open?id=1H_cWJVJd2e71Pw-zZNdEpVUPsG01UqeF) | [log](https://drive.google.com/open?id=1yIWcbV6sctGm-RkjX8jlsLFDI4_Y8_Ej) | [pred](https://drive.google.com/open?id=19NvAjO7Piy7180YC18QsV5EqjdJZevvd) |
| Rnd              | CS              | TEMP       | temp_vog_gt5_rand_samp_4Mar20          | [model](https://drive.google.com/open?id=14mNAtwcli0tx95KG1yEQT3UuGttpfwZI) | [log](https://drive.google.com/open?id=1LDCXngoLUG086l2GpPjkujjoycII0arc) | [pred](https://drive.google.com/open?id=1gpAm7-fK2-LF-RQSLHD5eVV8Xs2BVBqh) |
| Rnd              | CS              | SPAT       | spat_vog_gt5_rand_samp_4Mar20          | [model](https://drive.google.com/open?id=1m50h8OE4cUKgDy8BQyJnWjhuV1rgp5P5) | [log](https://drive.google.com/open?id=1Gz6nUU-VKYhQrGgnc27s-kfGgnADnrO7) | [pred](https://drive.google.com/open?id=1_wukiifDPVZPhs1-bT4MqJg-Wh_cbiO-) |
| CS+Rnd           | CS              | SEP        | sep_vog_p100_6Mar20                    | [model](https://drive.google.com/open?id=1V2hnY4mOVN5h9OHacz0gt8KlWaJ1Na7B) | [log](https://drive.google.com/open?id=1BTMgsRhgqiwFkbRB-WrpmfcGYSWFaWv2) | [pred](https://drive.google.com/open?id=10RLaf09-GS0mnFBbT6uIKVDLJyFz4Aym) |
| CS+Rnd           | CS              | TEMP       | temp_vog_3Mar20                        | [model](https://drive.google.com/open?id=1KGq0x4X3reRguvrtnxJ4YvMcOi2lDpsv) | [log](https://drive.google.com/open?id=12RLUr4037k33fzQjEfd6m4IU52IFizv_) | [pred](https://drive.google.com/open?id=1zT8XZxNGMD9L0GuSM--XLGAJDFsIwhjO) |
| CS+Rnd           | CS              | SPAT       | spat_vog_3Mar20                        | [model](https://drive.google.com/open?id=1jV3GhJLCDcHGDkj2NHUaWCz5RClwxoN7) | [log](https://drive.google.com/open?id=1_TRlAkqaYL0zg0zYnMmDMf_QkawOvZPr) | [pred](https://drive.google.com/open?id=1OaXFKF7ZhR8ziFbL088Vtct6lb8ZXSI2) |
| CS+Rnd           | Rnd             | SEP        | sep_vog_gt5_train_cs_test_rand_6Mar20  | -                                                                           | [log](https://drive.google.com/open?id=14NgQCvkEUZNNnBFe0u63ydxEyxY2xAwG) | [pred](https://drive.google.com/open?id=1i5Aa7e6bZcyLBse93pplmGNprNP1grKz) |
| CS+Rnd           | Rnd             | TEMP       | temp_vog_gt5_train_cs_test_rand_6Mar20 | -                                                                           | [log](https://drive.google.com/open?id=12lUmfmFegzxUC1FlIoWf39DwH_dbvGJ4) | [pred](https://drive.google.com/open?id=18-32-7oZDfOoli65PlCt7HotFKGpKu3L) |
| CS+Rnd           | Rnd             | SPAT       | spat_vog_gt5_train_cs_test_rand_6Mar20 | -                                                                           | [log](https://drive.google.com/open?id=1HIOMflSP_FIi3Uj_tVouol6St9z6ViI_) | [pred](https://drive.google.com/open?id=1zi4bA8DXwzzJNlUOEVNkrwzgWG6dS9kB) |


## GT5 Models, SPAT, Num Vids 
Page 7, Table 6


|   Num Vids | ID                   | []()                                                                        | []()                                                                      | []()                                                                       |
|------------|----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
|          2 | spat_vog_2vid_4Mar20 | [model](https://drive.google.com/open?id=1aWtn87POX-cUuYrLlW9ojOrOmOkBLS1D) | [log](https://drive.google.com/open?id=10xcs8z22BLXjIRgAuYXDI7oUEhv72jHE) | [pred](https://drive.google.com/open?id=1NplADG30rGhrhK5WtBWZ-n8giSnp_o5m) |
|          3 | spat_vog_3vid_4Mar20 | [model](https://drive.google.com/open?id=1Aa1jsH1ecalrFaoB6orX-r1_-6-gv1kr) | [log](https://drive.google.com/open?id=1vKOyEow3Ga5Rd5Vm8nvbppRjEeGldI7a) | [pred](https://drive.google.com/open?id=1RxQCoatCjjZ7J-i9IiXtQZAS9Riinkdo) |
|          5 | spat_vog_5vid_4Mar20 | [model](https://drive.google.com/open?id=128Jd4N9tVEOH4jxEco3kHv_Ha2iCAbPf) | [log](https://drive.google.com/open?id=11TaAYTIWcgnxF9CdvgvskLgrVYQCWkj5) | [pred](https://drive.google.com/open?id=1UYQpWB360XDzaDLKQZ2jP1hvIoBPu8-9) |

## GT5 Models, SPAT, Model Ablation
Page 7, Table 7


| MDL Name                 | ID                                | []()                                                                        | []()                                                                      | []()                                                                       |
|--------------------------|-----------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| ImgGrnd                  | spat_igrnd_3Mar20                 | [model](https://drive.google.com/open?id=1uH_LLQRegugRA_z7QILDuauEauk7M_QG) | [log](https://drive.google.com/open?id=1KK3750SuC8Ra0qvLmpODuS-EOnAE9Coy) | [pred](https://drive.google.com/open?id=1Pn_MuBKHb5kvShHn2YU1mHmuLtyW2OY5) |
| ImgGrnd_otx              | spat_vgrnd_3Mar20                 | [model](https://drive.google.com/open?id=1zLfVG8Uu4n9s1_iFglVrez4-Rd_4Zy9I) | [log](https://drive.google.com/open?id=1nxWJb3vSMwe8xD4HmZRIHrKXbHtCanwl) | [pred](https://drive.google.com/open?id=1745DOGTbvvHg34Xkf6NKw_c3c5yOQAM_) |
| ImgGrnd_otx_rpe          | spat_vgrnd_rel_4Mar20             | [model](https://drive.google.com/open?id=12UgfHYbea3Jy-iGvl171sK5cyybHxXtv) | [log](https://drive.google.com/open?id=1_Y0GWFqN9FrqhkNJy6o4j5ubB8B5ziIh) | [pred](https://drive.google.com/open?id=1O_Mq57w8-K34qrYpdSFM6QxmHi2aX5xt) |
| ImgGrnd_mtx              | spat_vog_only_mul_4Mar20          | [model](https://drive.google.com/open?id=1x0wfllNr4wv9NQE9icK2X81zylRckNe5) | [log](https://drive.google.com/open?id=13I55VpsYcIYjpmC5eq7hb96ykTT4Kzoy) | [pred](https://drive.google.com/open?id=1mgm9L1yk-XpYfGFGpJyBwGLPMgCoN9Ch) |
| ImgGrnd_mtx_rpe          | spat_vog_only_mul_rel_4Mar20      | [model](https://drive.google.com/open?id=1qP04OHLIPk7rJmgJHrBVTCh33qC8PoXd) | [log](https://drive.google.com/open?id=1EeK1oO0yhaElILK0U8A1-yRl2OzUv_f9) | [pred](https://drive.google.com/open?id=1uWME8D0DyaWlSpZAmzz8V0hJdMEIQIpj) |
| ImgGrnd_3L6H             | spat_vgrnd_3L6H_4Mar20            | [model](https://drive.google.com/open?id=1NSV9Thj_PpR_cQf_XTArlQWFyl5WfF_h) | [log](https://drive.google.com/open?id=1j6C_sOnUTbQnx-hOlQY9h8Zb3DI7GnhR) | [pred](https://drive.google.com/open?id=1QA6qHBjkr06eR36z754ITuZyorWzqQAJ) |
| ImgGrnd_otx_mtx_rpe      | spat_vog_3Mar20                   | [model](https://drive.google.com/open?id=1jV3GhJLCDcHGDkj2NHUaWCz5RClwxoN7) | [log](https://drive.google.com/open?id=1_TRlAkqaYL0zg0zYnMmDMf_QkawOvZPr) | [pred](https://drive.google.com/open?id=1OaXFKF7ZhR8ziFbL088Vtct6lb8ZXSI2) |
| VOGNet_mtx_3L6H          | spat_vog_3L6H_6Mar20              | [model](https://drive.google.com/open?id=1fnLr1tLeLwdzpxSAy85N0fzQrhZRlz0n) | [log](https://drive.google.com/open?id=1NwToFHpr8BDfBRmOsKXHUyv3LwE1tRSv) | [pred](https://drive.google.com/open?id=1KdZnvuZH0m5DStP177HziO8DCJ0oy6EY) |
| VOGNet_mtx_3L6H_otx_3L6H | spat_vog_obj_3L6H_mul_3L6H_8Mar20 | [model](https://drive.google.com/open?id=1N4orJqFZDwIjQdSBUZXf8jWLkv1jeM_w) | [log](https://drive.google.com/open?id=1xJdz6kgbpLpZYH9FZBghFFBShbg4g0Jo) | [pred](https://drive.google.com/open?id=1rDbxtocU_r6AzdRRhYsVV7_utCMT0W3s) |
