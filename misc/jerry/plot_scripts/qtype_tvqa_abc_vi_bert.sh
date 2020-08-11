#!/bin/bash
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python /home/jumperkables/kable_management/mk8+-tvqa/mystuff/question_type.py \
    --action acc_by_type_plot \
    --lanecheck_path /home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_vi_bert/lanecheck_dict.pickle \
    --model VI_bert