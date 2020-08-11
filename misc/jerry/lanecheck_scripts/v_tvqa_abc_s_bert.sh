#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=tvqa_abc_s_bert
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/v_tvqa_abc_s_bert.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/mk8+-tvqa/mystuff/validate.py \
    --input_streams sub \
    --jobname=tvqa_abc_s_bert \
    --best_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_s_bert/best_valid.pth \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --lanecheck True \
    --bert default \
    --lanecheck_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_s_bert/lanecheck_dict.pickle \
    --test_bsz 32 \
    --dset=valid
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############