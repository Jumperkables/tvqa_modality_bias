#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --qos short
#SBATCH --job-name=tvqa_abc_svir_bert_soff
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert_soff.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python /home/jumperkables/kable_management/mk8+-tvqa/mystuff/validate.py \
    --input_streams sub vcpt imagenet regional \
    --jobname=tvqa_abc_svir_bert_soff \
    --best_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/best_valid.pth \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --bert default \
    --lanecheck True \
    --lanecheck_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict_soff.pickle \
    --test_bsz 32 \
    --disable_streams sub \
    --regional_topk 20 \
    --dset=test
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############