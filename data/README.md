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
|-- anet
    |-- fc6_feat_100rois
    |-- fc6_feat_5rois
    |-- rgb_motion_1d
    |-- frames_10frm
|-- anet_srl
    |-- trn_srl_annots.csv
    |-- trn_srl_dict_obj_to_ind.json
    |-- val_srl_annots.csv
    |-- val_srl_dict_obj_to_ind.json
```

It should ~550 gb of data !!(need to get the exact amount)

NOTE: Highly advisable to have the features in SSD; otherwise massive drop in speed!


## Details about the Data
Here, I have explained the data contents in 1-line.
For an in-depth overview of the construction, please refer to [docode](../dcode)

1. `fc6_feat_Xrois`: We have 10 frames, for each frame we get X rois. `X=100` is obtained from FasterRCNN trained on Visual Genome. `X=5` is obtained from `X=100` such that ground-truth annotations are included and the remaining are the top scoring boxes. The latter setting allows us to perform easy experimentations.
1. `rgb_motion_1d`: RGB and FLOW features for frames (1fps) of the video.
1. `frames_10frm`: Used only for visualization purposes.
1. `{trn/val}_srl_annots.csv`: The main annotation files required for grounding.
1. `{trn/val}_dict_obj_to_ind.json`: Dictionary mapping helpful for sampling contrastive examples.

## Annotation File Structure:
To Be Added