# Creating ActivityNet SRL (ASRL) from ActivityNet Captions (AC) and ActivityNet Entities (AE)

The code is for generating the dataset from the parent datasets.
If you just want to use use ASRL as a training bed, you can skip this. See [DATA_README](../data/README.md)

## Quick summary

Very briefly, the process is as follows:
1. Add semantic roles to captions in AC.
1. Obtain the bounding boxes and category names from AE for the relevant phrases.
1. Filter out some verbs like "is", "are", "complete", "begin"
1. Filter some SRL Arguments based on Frequency.
1. Get Training/Validation/Test videos.

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


1. Obtain correspondig bounding boxes from AE (<1min)
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

1. Use the Train/Val videos from AE to create Train/Val/Test videos for ASRL (~5-7 mins)
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
