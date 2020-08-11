#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=tvqa_abc_svir
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/v_tvqa_abc_svir.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python /home/jumperkables/kable_management/mk8+-tvqa/validate.py \
    --input_streams sub vcpt imagenet regional \
    --jobname=tvqa_abc_svir \
    --regional_topk 20 \
    --best_path /home/jumperkables/kable_management/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/best_valid.pth \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --lanecheck True \
    --device 0 \
    --lanecheck_path /home/jumperkables/kable_management/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle \
    --test_bsz 32 \
    --dset=test
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############