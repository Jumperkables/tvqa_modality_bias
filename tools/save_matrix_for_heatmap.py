__author__ = 'Jumperkables'
import sys, os
sys.path.insert(1, "..")
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/tvqa_modality_bias"))
#sys.path.insert(1, os.path.expanduser("~/kable_management/mk8+-tvqa"))
from utils import load_pickle, save_pickle, load_json, files_exist
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random

# Take the paths of two lanecheck paths and then compare them and return their overlap
def ioveru(d_path_0, d_path_1):
    if d_path_0 ==  d_path_1:
        return 1
    dict_0 = load_pickle(d_path_0)
    dict_1 = load_pickle(d_path_1)
    acc_0 = dict_0['acc']
    acc_1 = dict_1['acc']
    del dict_0['acc']
    del dict_1['acc']
    correct_d0 = []
    correct_d1 = []
    total = dict_0.keys()
    for qid, sub_dict in dict_0.items():
        check = sub_dict.popitem()[1]
        if check[5]==check[6]:
            correct_d0.append(qid)
    for qid, sub_dict in dict_1.items():
        check = sub_dict.popitem()[1]
        if check[5]==check[6]:
            correct_d1.append(qid)
    union = len(list(set(correct_d0).union(correct_d1)))
    intersection = len(list(set(correct_d0).intersection(correct_d1)))
    #import ipdb; ipdb.set_trace()
    return(intersection/union)

def answer_agreement(d_path_0, d_path_1):
    if d_path_0 ==  d_path_1:
        return 1
    dict_0 = load_pickle(d_path_0)
    dict_1 = load_pickle(d_path_1)
    acc_0 = dict_0['acc']
    acc_1 = dict_1['acc']
    del dict_0['acc']
    del dict_1['acc']
    #import ipdb; ipdb.set_trace()
    agree = []
    disagree = []
    for qid in dict_0.keys():
        check_0 = dict_0[qid].popitem()[1]
        check_1 = dict_1[qid].popitem()[1]
        if check_0[6]==check_1[6]:
            agree.append(qid)
        else:
            disagree.append(qid)

    
    #import ipdb; ipdb.set_trace()
    return(len(agree)/(len(agree)+len(disagree)))

# 
def answers_per_q(d_path_0):
    dict_0 = load_pickle(d_path_0)
    acc_0 = dict_0['acc']
    del dict_0['acc']
    #import ipdb; ipdb.set_trace()
    qid2a = {}
    for qid in dict_0.keys():
        check_0 = dict_0[qid].popitem()[1]
        qid2a[qid] = (check_0[5], check_0[6])

    #import ipdb; ipdb.set_trace()
    return(qid2a, acc_0)


# argparse options to run this to extract question types
# or to plot onto pie charts
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create a question type dictionary, or plot it')
    parser.add_argument('--action',  type=str,
                        default=None, choices=['iou', 'agree', 'popvote'],
                        help='create iou matrix and save it')
    args = parser.parse_args()
    if not args.action:
        sys.exit()

    jerry_subpath = '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/'
    ncc_subpath = '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/ncc/results/'
    paths = [
        ncc_subpath+'tvqa_abc_v/',
        ncc_subpath+'tvqa_abc_v_bert/',
        ncc_subpath+'tvqa_abc_i/',
        ncc_subpath+'tvqa_abc_i_bert/',
        jerry_subpath+'tvqa_abc_r/',
        jerry_subpath+'tvqa_abc_r_bert/',
        ncc_subpath+'tvqa_abc_vi/',
        ncc_subpath+'tvqa_abc_vi_bert/',
        jerry_subpath+'tvqa_abc_vir/',
        jerry_subpath+'tvqa_abc_vir_bert/',
        ncc_subpath+'tvqa_abc_s/',
        ncc_subpath+'tvqa_abc_s_bert/',
        ncc_subpath+'tvqa_abc_si/',
        ncc_subpath+'tvqa_abc_si_bert/',
        ncc_subpath+'tvqa_abc_svi/',
        ncc_subpath+'tvqa_abc_svi_bert/',
        jerry_subpath+'tvqa_abc_svir/',
        jerry_subpath+'tvqa_abc_svir_bert/'
    ]
    train_plot_paths = [path+'lanecheck_dict.pickle_train' for path in paths]
    valid_plot_paths = [path+'lanecheck_dict.pickle_valid' for path in paths]
    labels = [
        'V',
        'V',
        'I',
        'I',
        'R',
        'R',
        'VI',
        'VI',
        'VIR',
        'VIR',
        'S',
        'S',
        'SI',
        'SI',
        'SVI',
        'SVI',
        'SVIR',
        'SVIR'
    ]

    # Labels and plot paths from here
    label_matrix = np.chararray((len(labels), len(labels)), itemsize=19)
    for idx0, label0 in enumerate(labels):
        for idx1, label1 in enumerate(labels):
            label_matrix[idx0][idx1] = label0+'/'+label1

    if args.action == 'iou':
        train_acc_matrix = np.zeros((len(train_plot_paths), len(train_plot_paths)))
        valid_acc_matrix = np.zeros((len(valid_plot_paths), len(valid_plot_paths)))

        for idx0, path0 in enumerate(valid_plot_paths):
            for idx1, path1 in enumerate(valid_plot_paths):
                valid_acc_matrix[idx0][idx1] = ioveru(path0, path1)
        print(valid_acc_matrix)
        
        return_dict = {'keys': label_matrix, 'acc_matrix': valid_acc_matrix}
        save_pickle(return_dict, '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/validation_iou_matrix.pickle')
        save_pickle(labels, '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/validation_iou_matrix_labels.pickle')

    if args.action == 'agree':
        train_acc_matrix = np.zeros((len(train_plot_paths), len(train_plot_paths)))
        valid_acc_matrix = np.zeros((len(valid_plot_paths), len(valid_plot_paths)))

        for idx0, path0 in enumerate(valid_plot_paths):
            for idx1, path1 in enumerate(valid_plot_paths):
                valid_acc_matrix[idx0][idx1] = answer_agreement(path0, path1)
        print(valid_acc_matrix)
        
        return_dict = {'keys': label_matrix, 'acc_matrix': valid_acc_matrix}
        save_pickle(return_dict, '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/validation_agree_matrix.pickle')
        save_pickle(labels, '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/validation_agree_matrix_labels.pickle')
    
    if args.action == 'popvote':
        models = []
        for idx, path in enumerate(valid_plot_paths):
            models.append(answers_per_q(path))
        threshold_acc = 0.5
        models = [ model for model in models if model[1]<threshold_acc ]
        qid_popvote = {}
        ground_truths = {}
        check_model = models[0]
        for qid in check_model[0].keys():
            qid_popvote[qid] = [0,0,0,0,0]
            ground_truths[qid] = check_model[0][qid][0]

        for model in models:
            # ground_truth / prediction
            for qid, gt_pred in model[0].items():
                #ground_truth = gt_pred[0]
                prediction = int(gt_pred[1])
                qid_popvote[qid][prediction] += 1
        for qid in qid_popvote.keys():
            temp = qid_popvote[qid]
            temp = temp.index(max(temp))
            qid_popvote[qid]=temp

        correct = 0
        total = len(qid_popvote.keys())
        for qid in qid_popvote.keys():
            if qid_popvote[qid] == ground_truths[qid]:
                correct += 1
        print(correct*100/total)
