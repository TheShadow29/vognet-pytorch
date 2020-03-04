"""
Select the model, loss, eval_fn
"""
from mdl_vog import (
    ImgGrnd_SEP,
    ImgGrnd_TEMP,
    ImgGrnd_SPAT,
    VidGrnd_SEP,
    VidGrnd_TEMP,
    VidGrnd_SPAT,
    VOG_SEP,
    VOG_TEMP,
    VOG_SPAT,
    LossB_SEP,
    LossB_TEMP,
    LossB_SPAT
)

from eval_vsrl_corr import (
    EvaluatorDS4_Corr_SSJ1_Sep,
    EvaluatorDS4_Corr_SSJ1_Temporal,
    EvaluatorDS4_Corr_SSJ1_Spatial
)


def get_mdl_loss_eval(cfg):
    conc_type = cfg.ds.conc_type
    mdl_type = cfg.mdl.name
    if conc_type == 'sep' or conc_type == 'svsq':
        if mdl_type == 'igrnd':
            mdl = ImgGrnd_SEP
        elif mdl_type == 'vgrnd':
            mdl = VidGrnd_SEP
        elif mdl_type == 'vog':
            mdl = VOG_SEP
        else:
            raise NotImplementedError
        loss = LossB_SEP
        evl = EvaluatorDS4_Corr_SSJ1_Sep
    elif conc_type == 'temp':
        if mdl_type == 'igrnd':
            mdl = ImgGrnd_TEMP
        elif mdl_type == 'vgrnd':
            mdl = VidGrnd_TEMP
        elif mdl_type == 'vog':
            mdl = VOG_TEMP
        else:
            raise NotImplementedError
        loss = LossB_TEMP
        evl = EvaluatorDS4_Corr_SSJ1_Temporal
    elif conc_type == 'spat':
        if mdl_type == 'igrnd':
            mdl = ImgGrnd_SPAT
        elif mdl_type == 'vgrnd':
            mdl = VidGrnd_SPAT
        elif mdl_type == 'vog':
            mdl = VOG_SPAT
        else:
            raise NotImplementedError
        loss = LossB_SPAT
        evl = EvaluatorDS4_Corr_SSJ1_Spatial
    else:
        raise NotImplementedError

    return {
        'mdl': mdl,
        'loss': loss,
        'eval': evl
    }
