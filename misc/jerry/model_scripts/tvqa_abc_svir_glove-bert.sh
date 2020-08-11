#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=svir_bert-glove
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert-glove.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/mk8+-tvqa/main.py \
    --input_streams sub vcpt imagenet regional \
    --jobname=tvqa_abc_svir_bert-glove \
    --results_dir_base=/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert-glove \
    --modelname=tvqa_abc_bert_nofc_bert-glove \
    --lrtype radam \
    --bsz 32 \
    --bert default \
    --regional_topk 20 \
    --log_freq 800 \
    --test_bsz 32 \
    --lanecheck_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert-glove/lanecheck_dict.pickle
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############