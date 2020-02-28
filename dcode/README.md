# Creating ActivityNet SRL (ASRL) from ActivityNet Captions (AC) and ActivityNet Entities (AE)

The code is for generating the dataset from the parent datasets.
If you just want to use use ASRL as a training bed, you can skip this. See [DATA_README](../data/README.md)

## Quick summary

Very briefly, the process is as follows:
1. Add semantic roles to captions in AC.
1. Prepocess AE. In particular, resize all the proposals, ground-truth bounding boxes (this is
required for SPAT/TEMP).
1. Preprocess the features and choose only 5 groundtruths for GT5 setting.
1. Obtain the bounding boxes and category names from AE for the relevant phrases.
1. Filter out some verbs like "is", "are", "complete", "begin"
1. Filter some SRL Arguments based on Frequency.
1. Get Training/Validation/Test videos.
1. Do Contrastive Sampling and store the dictionary files for easier sampling during training.

## Preprocessing Steps

1. First download relevant files.
Optional: specify the data folder where it would be downloaded.
```
bash download_asrl_parent_ann.sh [save_point]
```
The folder should look like:
```
anet_srl_scratch
|-- anet_captions_all_splits.json (AC captions)
|-- anet_entities_test_1.json
|-- anet_entities_test_2.json
|-- anet_entities_val_1.json
|-- anet_entities_val_2.json
|-- cap_anet_trainval.json        (AE Train annotations)
|-- dic_anet.json                 (Train/Valid/Test video splits for AE)
```

1. Use SRL Labeling system from AllenAI (Should take ~15 mins) to
add the semantic roles to the captions from AC.

```
cd $ROOT
python dcode/sem_role_labeller.py
```

This will create `$ROOT/cache_dir` and store the output SRL files which should look like:
```
cache_dir/
|-- SRL_Anet
    |-- SRL_Anet_bert_cap_annots.csv  # AC annotations in csv format to input into BERT
    |-- srl_bert_preds.pkl            # BERT outputs
```

1. Resize the boxes in AE.
```
cd $ROOT
python dcode/preproc_anet_files.py --task='resize_boxes_ae'
```
This takes the file `cap_anet_trainval.json` as input (this is the main AE annotation file) and outputs `anet_ent_cls_bbox_trainval.json`. The latter file contains resized ground-truth boxes.
It also resizes the proposal boxes, taking in `anet_detection_vg_fc6_feat_100rois.h5` as input and produces `anet_detection_vg_fc6_feat_100rois_resized.h5` as output. The latter contains resized proposals.

1. GT5 setting
```
cd $ROOT
python dcode/preproc_anet_files.py --task='choose_gt_5'
```
Intially, there are `100` proposals per frame.
For faster iteration, we only choose the 5 proposals from each frame.
If there is a ground-truth box, we take include that box, and the remaining are included in order of their proposal score (not a fair way, but the best that could be done).
If there are no ground-turth box, we choose the top5 scoring proposals.

To compute the recall scores (for sanity check):
```
python dcode/preproc_anet_files.py --task='compute_recall'
```
By default, it computes recall scores for GT5, you can change the proposal file, for other settings.

1. Aligning SRL outputs and NounPhrases from AE to create ASRL and adding the bounding boxes to the ASRL files (<1min)
```
cd $ROOT
python dcode/asrl_creator.py
```
Now `$ROOT/data/anet_verb/` should look like:
```
anet_verb/
|-- verb_ent_file.csv        # main file with SRLs, BBoxes
|-- verb_lemma_dict.json     # dictionary of verbs corresponding to their lemma
```

1. Use the Train/Val videos from AE to create Train/Val/Test videos for ASRL (~5-7 mins).
Additionally, create the vocab file for the SRL arguments
```
cd $ROOT
python dcode/prepoc_ds_files.py
```
This will create `anet_srl_scratch/csv_dir`. It should look like:
```
csv_dir
|-- train.csv
|-- train_postproc.csv
|-- val.csv
|-- val_postproc.csv
```

Further, now `$ROOT/data/anet_verb/` should look like:
```
anet_verb/
|-- trn_verb_ent_file.csv       # train file
|-- val_verb_ent_file.csv       # val & test file
|-- verb_ent_file.csv
|-- verb_lemma_dict.json
```

1. Do Constrastive sampling for train and validation set (~30mins)
```
cd $ROOT
python code/ds4_creator.py
```

Now your `anet_verb` directory should look like:
```
anet_verb/
|-- trn_srl_annots_with_ds4_inds.csv     # used for training
|-- trn_srl_args_dict_obj_to_ind.json    # used for CS
|-- trn_verb_ent_file.csv                # not used anymore
|-- val_srl_annots_with_ds4_inds.csv     # used for val/test
|-- val_srl_args_dict_obj_to_ind.json    # used for CS
|-- val_verb_ent_file.csv                # not used anymore
|-- verb_ent_file.csv                    # not used anymore
|-- verb_lemma_dict.json                 # not used anymore
```
