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
|-- anet_cap_ent_files (31M)
    |-- anet_captions_all_splits.json
    |-- anet_ent_cls_bbox_trainval.json
    |-- csv_dir
	 |-- train.csv
	 |-- train_postproc.csv
	 |-- val.csv
	 |-- val_postproc.csv
    |-- dic_anet.json
|-- anet_srl_files (112M)
    |-- arg_vocab.pkl
    |-- trn_asrl_annots.csv
    |-- trn_srl_obj_to_index_dict.json
    |-- val_asrl_annots.csv
    |-- val_srl_obj_to_index_dict.json
```

It should ~530 gb of data !!

NOTE: Highly advisable to have the features in SSD; otherwise massive drop in speed!


## Details about the Data
Here, I have explained the data contents in 1-line.
For an in-depth overview of the construction, please refer to [DATA PREP README](../dcode/README.md)

1. `fc6_feat_Xrois`: We have 10 frames, for each frame we get X rois. `X=100` is obtained from FasterRCNN trained on Visual Genome. `X=5` is obtained from `X=100` such that ground-truth annotations are included and the remaining are the top scoring boxes. The latter setting allows us to perform easy experimentations.
1. `rgb_motion_1d`: RGB and FLOW features for frames (1fps) of the video.
1. `{trn/val}_asrl_annots.csv`: The main annotation files required for grounding.
1. `{trn/val}_srl_obj_to_index_dict.json`: Dictionary mapping helpful for sampling contrastive examples.

## Annotation File Structure:
The main annotation files for ASRL are `{trn/val}_asrl_annots.csv`

We use Video Segments of the ActivityNet since we are focussing on Trimmed videos only.

ActivityNet Entities provides the bounding boxes for the noun-phrases in ActivityNet Captions. For more details please refer to [dcode](../dcode)

`trn_asrl_annots.csv` has 26 columns!

Lets consider the first example. You can get this using:
```
import pandas as pd
trn_csv = pd.read_csv('./trn_asrl_annots.csv')
first_data_point = trn_csv.iloc[0]
column_list = ['srl_ind', 'vid_seg']
```

1. `srl_ind`: the index in this csv file. Here it is `0`
1. `vt_split`: is the split the data point belongs to. All data points in `trn_asrl_anonts.csv` have this set to `train`. However, it is 50-50 split for `val_asrl_annots.csv` for `val` and `test`.
1. `vid_seg`: the video and the segment of the video the file belongs to. The convention used is `{vid_name}_segment_{seg_id:02d}`. Here it is `v_--0edUL8zmA_segment_00` which means, it is the 0th segment of the video `v_--0edUL8zmA`.
1. `ann_ind`: this is the index in the `anet_cap_ent_files/csv_dir/{trn/val}_postproc.csv` file. This index is used to retrieve the proposal boxes from `anet_detection_vg_fc6_feat_100rois_resized.h5`. Here it is `28557` which means 28557th row of the h5 file corresponds to this `vid_seg`.
1. `sent`: this is the main sentence provided in the activitynet captions for the given vid_seg. The sentence may contain multiple verbs, and as such data points sharing the same vid seg will have the same sentence. Here, the sentence is "Four men are playing dodge ball in an indoor court ."
1. `words`: this is simply tokenization of `sent`. Here it is: \['Four', 'men', 'are', 'playing', 'dodge', 'ball', 'in', 'an', 'indoor', 'court', '.'\]
1. `verb`: we pass the sentence through a semantic role labeler (see [demo](https://demo.allennlp.org/semantic-role-labeling)) which extracts multiple verbs from the sentence and assigning semantic roles pivoted for each verb. Each verb is treated as a separate data point. Here, the verb is `playing`.
1. `tags`: The BIO tagging output from the SRL for the given verb. Here it is \['B-ARG0', 'I-ARG0', 'O', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-LOC', 'I-ARGM-LOC', 'I-ARGM-LOC', 'I-ARGM-LOC', 'O'\] which basically the structure "playing: \[ARG0: Four men] are \[V: playing] \[ARG1: dodge ball] \[ARGM-LOC: in an indoor court] ."
1. `req_pat_ix`: Same information as `tags` but represented as List\[Tuple\[ArgX, List\[word indices]]. The word indices correspond to the output of `word`. Here it is `[['ARG0', [0, 1]], ['V', [3]], ['ARG1', [4, 5]], ['ARGM-LOC', [6, 7, 8, 9]]]` which suggests `word[0], word[1]` constitute ARG0 (basically \[ARG0: Four men])
1. `req_pat`: Same information as above, just the list of word indices are replaced with space separated words. Here it is:  \[('ARG0', 'Four men'), ('V', 'playing'), ('ARG1', 'dodge ball'), ('ARGM-LOC', 'in an indoor court')]
1. `req_aname`: Same as `req_pat` just that it only extracts the words without the argument roles. Here it is: \['Four men', 'playing', 'dodge ball', 'in an indoor court']
1. `req_args`: Instead of the words, only stores the semantic roles. Here it is \['ARG0', 'V', 'ARG1', 'ARGM-LOC']
1. `gt_bboxes`: The ground-truth boxes (4d) provided in AE for the given vid-seg. It is List\[List\[x1,y1,x2,y2]]
1. `gt_frms`: The frames (ranging from 0-9) where they are annotated. It is List\[\len(gt_bboxes)]
1. `process_idx2`: It provides the word index for the given bounding box. It is List\[List\[int]]. Here it is `[[1], [1], [1], [1], [9]]`. Note that `word[1] = men` which means the first four bounding boxes refer to the four men and the final bounding box refers to the `court`.
1. `process_clss`: Lemmatized Noun for the words in `process_idx2`. Here it is `[['man'], ['man'], ['man'], ['man'], ['court']]`
1. `req_cls_pats`: Same as `req_pat` with the words replaced with their lemmatized noun. `[('ARG0', ['man']), ('V', ['playing']), ('ARG1', ['dodge', 'ball']), ('ARGM-LOC', ['court'])]`
1. `req_cls_pats_mask`: It is List\[Tuple\[ArgX, Mask, GTBox Index list]]. ArgX is the Argument Name like Arg0, Mask = 1 means this role has a bounding box, 0 implies the role doesn't have a bounding box and hence is not evaluated. GTBox Index List is the list of indices of the bounding boxes refering to this role. Here it is `[('ARG0', 1, [0, 1, 2, 3]), ('V', 0, [0]), ('ARG1', 0, [0]), ('ARGM-LOC', 1, [4])]` which implies ARG0 and ARGM-LOC are groundable, while V and ARG1 are not. Moreover, the first four bounding boxes refer to ARG0 and the last bounding box refers to ARGM-LOC.
1. `lemma_ARGX`: The lemmatized verb/argument role used for contrastive sampling.
1. `DS4_Inds`: For each role, it contains indices for which everything other than the lemmatized word for the argument role is same.
1. `ds4_msk`: If such contrastive samples were successfully found.
1. `RandDS4_Inds`: Simply random indices.
