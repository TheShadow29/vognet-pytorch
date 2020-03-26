#!/bin/bash
# Downloading script

CUR_DIR=$(pwd)
DATA_ROOT=${2:-CUR_DIR}

mkdir -p $DATA_ROOT/anet

function gdrive_download () {
	CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
	rm -rf /tmp/cookies.txt
}

function asrl_ann_dwn(){
    echo "Downloading ActivityNet SRL annotations"
    gdrive_download 1qSsD3AbWqw-KNObNg6N8xbTnF-Bg_eZn anet_verb.zip
    unzip anet_verb.zip && rm anet_verb.zip

    gdrive_download 1aZyLNP-VXS3stZpenWMuCTRF_NL2gznu anet_srl_scratch.zip
    unzip anet_srl_scratch.zip && rm anet_srl_scratch.zip
    echo "Saved Folder"
}

function anet_feats_dwn(){
    echo "Downloading ActivityNet Feats. May take some time"
    # Courtesy of Louwei Zhou, obtained from the repository:
    # https://github.com/facebookresearch/grounded-video-description/blob/master/tools/download_all.sh
    cd $DATA_ROOT/anet
    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz
    tar -xvzf rgb_motion_1d.tar.gz && rm rgb_motion_1d.tar.gz

    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_detection_vg_fc6_feat_100rois.h5

    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz
    tar -xvzf fc6_feat_100rois.tar.gz && rm fc6_feat_100rois.tar.gz

    gdrive_download 13tvBIEAgv4VS5dqkZBK1gvTI_Z22gRLM fc6_feat_5rois.zip
    unzip fc6_feat_5rois.zip && rm fc6_feat_5rois.zip

    gdrive_download 1a6UOK90Epz7n-dncKAeFDQP4TBgqdTS9 anet_detn_proposals_resized.zip
    unzip anet_detn_proposals_resized.zip && rm anet_detn_proposals_resized.zip
    cd $CUR_DIR
}

function dwn_all(){
    asrl_ann_dwn
    anet_feats_dwn
}


if [ "$1" = "asrl_anns" ]
then
    asrl_ann_dwn

elif [ "$1" = "anet_feats" ]
then
    anet_feats_dwn
elif [ "$1" = "all" ]
then
    dwn_all
else
    echo "Failed: Use download_data.sh asrl_anns | anet_feats | all"
    exit 1
fi
