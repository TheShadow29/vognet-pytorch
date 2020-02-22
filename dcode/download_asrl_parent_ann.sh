#!/bin/bash
# Downloading script

# CUR_DIR=$(pwd)
DDIR=${1:-"./data"}
DATA_ROOT=$DDIR/anet_srl_scratch

echo $DATA_ROOT
mkdir -p $DATA_ROOT

function ac_ae_dwn(){
    echo "Downloading ActivityNet Captions and ActivityNet Entities"
    # Courtesy of Louwei Zhou, obtained from the repository:
    # https://github.com/facebookresearch/grounded-video-description/blob/master/tools/download_all.sh
    # https://github.com/facebookresearch/ActivityNet-Entities/blob/master/data/anet_entities_cleaned_class_thresh50_trainval.json
    wget -P $DATA_ROOT https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz
    wget -P $DATA_ROOT https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz
    wget -P $DATA_ROOT https://raw.githubusercontent.com/facebookresearch/ActivityNet-Entities/master/data/anet_entities_cleaned_class_thresh50_trainval.json
    cd $DATA_ROOT
    tar -xvzf anet_entities_prep.tar.gz && rm anet_entities_prep.tar.gz
    tar -xvzf anet_entities_captions.tar.gz && rm anet_entities_captions.tar.gz
    cd $CUR_DIR
    echo "Finished downloading ActivityNet Captions and ActivityNet Entities"
}

ac_ae_dwn
