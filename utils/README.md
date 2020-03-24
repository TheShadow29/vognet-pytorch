# File Organization

1. `box_utils.py` as the name duly suggest, contains utils for box iou, box area computation. Also, contains code for box iou for multiple frames
1. `mdl_srl_utils.py` has convenience functions for the models (surprise surprise). Stuff like LSTM implementation adapted from fairseq.
1. `trn_utils.py` contains learner which handles the model saving/loading, logging stuff, saving predictions among other things.
