#!/bin/bash
source /home/crhf63/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/crhf63/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub imagenet \
    --jobname=blp_tvqa_2stream_glove_mcb_1500 \
    --results_dir_base=/home/crhf63/kable_management/blp_paper/.results/tvqa/2stream_glove_mcb_1500 \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 8 \
    --log_freq 3200 \
    --test_bsz 50 \
    --lanecheck_path /home/crhf63/kable_management/blp_paper/.results/tvqa/2stream_glove_mcb_1500/lanecheck_dict.pickle \
    --pool_type MCB \
    --dual_stream \
    --pool_out_dim 750
 
