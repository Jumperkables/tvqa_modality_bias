#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=tvqa_abc_svi
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/shap_tvqa_abc_svi.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python /home/jumperkables/kable_management/mk8+-tvqa/mystuff/shappy.py \
    --input_streams sub vcpt imagenet \
    --jobname=shap_tvqa_abc_si \
    --best_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svi/best_valid.pth \
    --modelname=tvqa_abc_bert_nofc_shap \
    --lrtype radam \
    --bsz 32 \
    --device 1 \
    --test_bsz 32
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############