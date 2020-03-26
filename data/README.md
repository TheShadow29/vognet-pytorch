# Preparing Data

This part is to download the data and start with the experiments.

If instead you are interested in generating ActivityNet-SRL from scratch (not required in general), see [dcode](../dcode).

## Quickstart

Optional: set the data folder.
```
cd $ROOT/data
bash download_data.sh all [data_folder]
```

After everything is downloaded successfully, the folder structure should look like:

```
data
|-- anet (530gb)
    |-- anet_detection_vg_fc6_feat_100rois.h5
    |-- anet_detection_vg_fc6_feat_100rois_resized.h5
    |-- anet_detection_vg_fc6_feat_gt5_rois.h5
    |-- fc6_feat_100rois
    |-- fc6_feat_5rois
    |-- rgb_motion_1d
    |-- vid_hw_dict.json
|-- anet_srl_scratch (79M)
    |-- anet_captions_all_splits.json
    |-- anet_ent_cls_bbox_trainval.json
    |-- anet_entities_cleaned_class_thresh50_trainval.json
    |-- anet_entities_prepocessed_clss_trainval.json
    |-- anet_entities_test_1.json
    |-- anet_entities_test_2.json
    |-- anet_entities_val_1.json
    |-- anet_entities_val_2.json
    |-- cap_anet_trainval.json
    |-- csv_dir
        |-- train.csv
        |-- train_postproc.csv
        |-- val.csv
        |-- val_postproc.csv
    |-- dic_anet.json
|-- anet_verb (260M)
    |-- arg_vocab.pkl
    |-- trn_srl_annots_with_ds4_inds.csv
    |-- trn_srl_args_dict_obj_to_ind.json
    |-- trn_verb_ent_file.csv
    |-- val_srl_annots_with_ds4_inds.csv
    |-- val_srl_args_dict_obj_to_ind.json
    |-- val_verb_ent_file.csv
    |-- verb_ent_file.csv
    |-- verb_lemma_dict.json
```

It should ~530 gb of data !!(need to get the exact amount)

NOTE: Highly advisable to have the features in SSD; otherwise massive drop in speed!


## Details about the Data
Here, I have explained the data contents in 1-line.
For an in-depth overview of the construction, please refer to [DATA PREP README](../dcode/README.md)

1. `fc6_feat_Xrois`: We have 10 frames, for each frame we get X rois. `X=100` is obtained from FasterRCNN trained on Visual Genome. `X=5` is obtained from `X=100` such that ground-truth annotations are included and the remaining are the top scoring boxes. The latter setting allows us to perform easy experimentations.
1. `rgb_motion_1d`: RGB and FLOW features for frames (1fps) of the video.
1. `{trn/val}_srl_annots.csv`: The main annotation files required for grounding.
1. `{trn/val}_dict_obj_to_ind.json`: Dictionary mapping helpful for sampling contrastive examples.

## Annotation File Structure:
To Be Added
