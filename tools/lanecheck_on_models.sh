#!/bin/bash
#source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python /home/jumperkables/kable_management/mk8+-tvqa/mystuff/lanecheck_on_models.py \
    --l_path=/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_valid \
    --input_streams sub vcpt imagenet regional \
    --off_streams sub \
    --regional_topk 20 
