#!/bin/bash
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub imagenet \
    --jobname=blp_tvqa_2stream_bert_mcb_1500 \
    --results_dir_base=/home/jumperkables/temp/2stream_bert_mcb_1500 \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 8 \
    --log_freq 3200 \
    --test_bsz 50 \
    --lanecheck_path /home/jumperkables/temp/lanecheck_dict.pickle \
    --pool_type MCB \
    --bert default \
    --dual_stream \
    --no_core_driver \
    --pool_out_dim 750

