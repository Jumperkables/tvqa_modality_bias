#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=svr
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svr.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/mk8+-tvqa/main.py \
    --input_streams sub vcpt regional \
    --jobname=tvqa_abc_svr \
    --results_dir_base=/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svr \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --regional_topk 20 \
    --log_freq 800 \
    --test_bsz 32 \
    --lanecheck_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svr/lanecheck_dict.pickle
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############kable_management/mk8+-tvqa/dataset_paper/jerry/results