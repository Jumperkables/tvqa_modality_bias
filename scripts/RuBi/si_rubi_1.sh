#!/bin/bash
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub imagenet \
    --jobname=tvqa_si_rubi_1 \
    --results_dir_base=/home/jumperkables/si_rubi_1 \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 32 \
    --lanecheck_path /home/jumperkables/si_rubi_1/lanecheck_dict.pickle \
    --rubi \
    --lanecheck True \
    --rubi_qloss_weight 1

