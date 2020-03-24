#!/bin/bash
# Downloading script

CUR_DIR=$(pwd)
DATA_ROOT=${2:-CUR_DIR}

mkdir -p $DATA_ROOT/anet_srl
mkdir -p $DATA_ROOT/anet
mkdir -p $DATA_ROOT/anet_verb

function asrl_ann_dwn(){
    echo "Downloading ActivityNet SRL annotations"
    wget -P $DATA_ROOT/anet_srl -c xxx.zip
    unzip xxx.zip && rm xxx.zip
    echo "Saved Folder"
}

function anet_feats_dwn(){
    echo "Downloading ActivityNet Feats. May take some time"
    # Courtesy of Louwei Zhou, obtained from the repository:
    # https://github.com/facebookresearch/grounded-video-description/blob/master/tools/download_all.sh
    wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz
    wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_detection_vg_fc6_feat_100rois.h5
    wget -P $DATA_ROOT/anet/ https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz
    cd $DATA_ROOT/anet
    tar -xvzf rgb_motion_1d.tar.gz && rm rgb_motion_1d.tar.gz
    tar -xvzf fc6_feat_100rois.tar.gz && rm fc6_feat_100rois.tar.gz
    cd $CUR_DIR
}

function dwn_all(){
    asrl_ann_dwn
    anet_feats_dwn
}

function symlnk(){
    ln -s $DATA_ROOT $CUR_DIR
}

if [ "$1" = "anet_srl_anns" ]
then
    asrl_ann_dwn

elif [ "$1" = "anet_feats" ]
then
    anet_feats_dwn
elif [ "$1" = "all" ]
then
    dwn_all
else
    echo "Failed: Use download_data.sh anet_srl_anns | anet_feats | all"
    exit 1
fi

symlnk
