#!/bin/bash
# Downloading script

CUR_DIR=$(pwd)
DDIR=${2:-"../data"}
DATA_ROOT=$DDIR/anet_cap_ent_files

echo $DATA_ROOT
mkdir -p $DDIR/anet_srl_files
mkdir -p $DATA_ROOT

function gdrive_download () {
	CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
	rm -rf /tmp/cookies.txt
}

function anet_feats_dwn(){
    echo "Downloading ActivityNet Feats. May take some time"
    # Courtesy of Louwei Zhou, obtained from the repository:
    # https://github.com/facebookresearch/grounded-video-description/blob/master/tools/download_all.sh
    cd $DDIR/anet
    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/rgb_motion_1d.tar.gz
    tar -xvzf rgb_motion_1d.tar.gz && rm rgb_motion_1d.tar.gz

    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_detection_vg_fc6_feat_100rois.h5

    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/fc6_feat_100rois.tar.gz
    tar -xvzf fc6_feat_100rois.tar.gz && rm fc6_feat_100rois.tar.gz

    # gdrive_download 13tvBIEAgv4VS5dqkZBK1gvTI_Z22gRLM fc6_feat_5rois.zip
    # unzip fc6_feat_5rois.zip && rm fc6_feat_5rois.zip

    # gdrive_download 1a6UOK90Epz7n-dncKAeFDQP4TBgqdTS9 anet_detn_proposals_resized.zip
    # unzip anet_detn_proposals_resized.zip && rm anet_detn_proposals_resized.zip
    cd $CUR_DIR
}

function ac_ae_dwn(){
    echo "Downloading ActivityNet Captions and ActivityNet Entities"
    cd $DATA_ROOT
    # Courtesy of Louwei Zhou, obtained from the repository:
    # https://github.com/facebookresearch/grounded-video-description
    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_prep.tar.gz
    wget https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/anet_entities_captions.tar.gz
    wget https://raw.githubusercontent.com/facebookresearch/ActivityNet-Entities/master/data/anet_entities_cleaned_class_thresh50_trainval.json
    tar -xvzf anet_entities_prep.tar.gz && rm anet_entities_prep.tar.gz
    tar -xvzf anet_entities_captions.tar.gz && rm anet_entities_captions.tar.gz
    cd $CUR_DIR
    echo "Finished downloading ActivityNet Captions and ActivityNet Entities"
}

function processed_files_dwn(){
    echo "Downloading ASRL processed files"
    cd $DDIR
    mkdir asrl_processed_files
    cd asrl_processed_files
    gdrive_download "1mH8TyVPU4w7864Hxiukzg8dnqPIyBuE3" anet_srl_files_all.zip
    gdrive_download "1vGgqc8_-ZBk3ExNroRP-On7ArWN-d8du" SRL_Anet.zip
    gdrive_download "1a6UOK90Epz7n-dncKAeFDQP4TBgqdTS9" anet_detn_proposals_resized.zip
    # gdrive_download "13tvBIEAgv4VS5dqkZBK1gvTI_Z22gRLM" fc6_feat_5rois.zip
    cd $CUR_DIR
}

function dwn_all(){
    ac_ae_dwn
    anet_feats_dwn
}

if [ "$1" = "ac_ae_anns" ]
then
    ac_ae_dwn
elif [ "$1" = "anet_feats" ]
then
    anet_feats_dwn
elif [ "$1" = "asrl_proc_files" ]
then
    processed_files_dwn
elif [ "$1" = "all" ]
then
    dwn_all
else
    echo "Failed: Use download_asrl_parent_ann.sh ac_ae_anns | anet_feats | asrl_proc_files | all"
    exit 1
fi
