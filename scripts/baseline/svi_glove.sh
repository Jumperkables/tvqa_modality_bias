#!/bin/bash
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate

python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=svi_glove \
    --results_dir_base=/home/jumperkables/temp/svi_glove \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 100 \
    --lanecheck True \
    --lanecheck_path /home/jumperkables/temp/svi_glove/lanecheck_dict.pickle 
