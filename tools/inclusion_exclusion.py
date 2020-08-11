import sys, os
#import torch
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/tvqa_modality_bias"))
sys.path.insert(1, "..")
from utils import save_pickle, load_pickle, load_json, files_exist
import argparse
import numpy as np

maxx = 0
minn = 0

# A list of paths to lanecheck dictionaries you wish to find the QID's of which questions are in or not in
def in_notin(model_dict, qid_list, in_list, not_in_list, validortrain):   # For convenience, leaving in list empty will assume all are to be considered
    import copy
    # Full QID list
    in_qids = []
    not_in_qids = []
    # For all models in the in_list
    for model in in_list:
        in_qids += model_dict[model]
    for model in not_in_list:
        not_in_qids += model_dict[model]

    # Unique QIDs
    in_qids = set(list(set(in_qids)))
    not_in_qids = set(list(set(not_in_qids)))
    in_qids = list(in_qids - not_in_qids)
    #in_qids = [qid for qid in in_qids if qid not in not_in_qids]
    print(round(len(in_qids)*100/len(qid_list),2), in_list, not_in_list, validortrain )
    return(in_qids)

def popular_vote(model_dict, qid_list, listy, validortrain):   # For convenience, leaving in list empty will assume all are to be considered
    # Full QID list
    listy = []
    import pdb; pdb.set_trace()
    # For all models in the in_list
    for model in listy:
        in_qids += model_dict[model]
    for model in not_in_list:
        not_in_qids += model_dict[model]

    # Unique QIDs
    in_qids = set(list(set(in_qids)))
    not_in_qids = set(list(set(not_in_qids)))
    in_qids = list(in_qids - not_in_qids)
    #in_qids = [qid for qid in in_qids if qid not in not_in_qids]
    print(round(len(in_qids)*100/len(qid_list),2), in_list, not_in_list, validortrain )
    return(in_qids)
    

def correct_on_qid_streams(opts):
    path, on_streams, off_streams = opts
    dicty = load_pickle(path)
    del dicty['acc']
    correct_qids = []

    # Initialise all streams
    vcpt_flag = False
    sub_flag = False
    imagenet_flag = False
    regtopk_flag = False

    # Turn on all relevant streams
    if 'vcpt' in on_streams:
        vcpt_flag = True
    if 'sub' in on_streams:
        sub_flag = True
    if 'imagenet' in on_streams:
        imagenet_flag = True
    if 'regional' in on_streams:
        regtopk_flag = True

    # Turn off all disabled streams
    if 'vcpt' in off_streams:
        vcpt_flag = False
    if 'sub' in off_streams:
        sub_flag = False
    if 'imagenet' in off_streams:
        imagenet_flag = False
    if 'regional' in off_streams:
        regtopk_flag = False

    # Aggregate all responses
    for qid, q_dict in dicty.items():
        answers = [0]*5
        try:
            ground_truth = q_dict['vcpt_out'][5]
        except:
            try:
                ground_truth = q_dict['sub_out'][5]
            except:
                try:
                    ground_truth = q_dict['vid_out'][5]
                except:
                    ground_truth = q_dict['regtopk_out'][5]
        if vcpt_flag:
            answers[0] += q_dict['vcpt_out'][0]
            answers[1] += q_dict['vcpt_out'][1]
            answers[2] += q_dict['vcpt_out'][2]
            answers[3] += q_dict['vcpt_out'][3]
            answers[4] += q_dict['vcpt_out'][4]
        if sub_flag:
            answers[0] += q_dict['sub_out'][0]
            answers[1] += q_dict['sub_out'][1]
            answers[2] += q_dict['sub_out'][2]
            answers[3] += q_dict['sub_out'][3]
            answers[4] += q_dict['sub_out'][4]
        if imagenet_flag:
            answers[0] += q_dict['vid_out'][0]
            answers[1] += q_dict['vid_out'][1]
            answers[2] += q_dict['vid_out'][2]
            answers[3] += q_dict['vid_out'][3]
            answers[4] += q_dict['vid_out'][4]
        if regtopk_flag:
            answers[0] += q_dict['regtopk_out'][0]
            answers[1] += q_dict['regtopk_out'][1]
            answers[2] += q_dict['regtopk_out'][2]
            answers[3] += q_dict['regtopk_out'][3]
            answers[4] += q_dict['regtopk_out'][4]
        
        # The predicted answer from all wanted lanes
        guess = answers.index(max(answers)) ####HERE
        if guess == ground_truth:
            correct_qids.append(qid)
    return(correct_qids)

def streams(stringy):
    streams = []
    if 'V' in stringy:
        streams.append('vcpt')
    if 'I' in stringy:
        streams.append('imagenet')
    if 'R' in stringy:
        streams.append('regional')
    if 'S' in stringy:
        streams.append('sub')
    return streams

def create_correct_qid_matrix():
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
    
    # Glove Train
    glove_train_opts = [ (train_plot_paths[i], streams(labels[i]), [] ) for i in range(len(labels)) if i%2 == 0 ]
    glove_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_train' , streams("SVIR") , streams("S") ) )
    glove_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_train' , streams("SVIR") , streams("V") ) )
    glove_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_train' , streams("SVIR") , streams("I") ) )
    glove_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_train' , streams("SVIR") , streams("R") ) )
    glove_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_train' , streams("SVIR") , streams("VIR") ) ) 
    # Glove Valid
    glove_valid_opts = [ (valid_plot_paths[i], streams(labels[i]), [] ) for i in range(len(labels)) if i%2 == 0 ]
    glove_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("S") ) )
    glove_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("V") ) )
    glove_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("I") ) )
    glove_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("R") ) )
    glove_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("VIR") ) ) 
    # BERT Train
    bert_train_opts = [ (train_plot_paths[i], streams(labels[i]), [] ) for i in range(len(labels)) if i%2 == 1 ]
    bert_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_train' , streams("SVIR") , streams("S") ) )
    bert_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_train' , streams("SVIR") , streams("V") ) )
    bert_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_train' , streams("SVIR") , streams("I") ) )
    bert_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_train' , streams("SVIR") , streams("R") ) )
    bert_train_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_train' , streams("SVIR") , streams("VIR") ) ) 
    # BERT Valid
    bert_valid_opts = [ (valid_plot_paths[i], streams(labels[i]), [] ) for i in range(len(labels)) if i%2 == 1 ]
    bert_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("S") ) )
    bert_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("V") ) )
    bert_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("I") ) )
    bert_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("R") ) )
    bert_valid_opts.append( ( '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_valid' , streams("SVIR") , streams("VIR") ) ) 

    # Returning Correct QIDS
    extra_labels = ['SVIR S-off', 'SVIR V-off', 'SVIR I-off', 'SVIR R-off', 'SVIR VIR-off']
    labels = ['V','I','R','VI','VIR','S','SI','SVI','SVIR']
    labels = labels + extra_labels
    print("Starting")
    glove_train_correct_qids = {labels[i]:correct_on_qid_streams(glove_train_opts[i]) for i in range(len(labels))}
    print("Glove_train done")
    glove_valid_correct_qids = {labels[i]:correct_on_qid_streams(glove_valid_opts[i]) for i in range(len(labels))}
    print("Glove_valid done")
    bert_train_correct_qids = {labels[i]:correct_on_qid_streams(bert_train_opts[i]) for i in range(len(labels))}
    print("bert_train done")
    bert_valid_correct_qids = {labels[i]:correct_on_qid_streams(bert_valid_opts[i]) for i in range(len(labels))}
    print("bert_valid done")
    return glove_train_correct_qids, glove_valid_correct_qids , bert_train_correct_qids, bert_valid_correct_qids


def plot_qtype_heatmap_table(qids_matrix, qids_labels):
    global maxx
    global minn
    from heatmap_table import shiftedColorMap, heatmap, annotate_heatmap
    import matplotlib.pyplot as plt
    qtype_labels = ['(55.6%) What','(11.6%) Who','(10.4%) Why','(11.7%) Where','(9.0%) How','(0.7%) Which','(1.1%) Other']
    colours = plt.cm.bwr#normal(vals))
    colours = shiftedColorMap(colours, start=0, midpoint=1-(maxx/(maxx - minn)), stop=1, name='shifted')


    fig, ax = plt.subplots()
    #im, cbar
    im, cbar = heatmap(qids_matrix, qids_labels, qtype_labels, ax=ax,
                   cmap=colours  ,cbarlabel='%'+' Increase in Each Question Type') 
    texts = annotate_heatmap(im, valfmt="{x:.1f}%")

    fig.tight_layout()
    plt.ylabel('Subset Name')
    plt.show()

def qids_2_qtype_dist(qids):
    global maxx
    global minn
    qid2qtype = load_pickle(os.path.expanduser("~/kable_management/mk8+-tvqa/dataset_paper/val_qid2qtype.pickle"))
    average = load_pickle(os.path.expanduser("~/kable_management/mk8+-tvqa/dataset_paper/val_qtype_averages.pickle"))
    idx_dict = {'what':0, 'who':1, 'why':2, 'where':3, 'how':4, 'which':5, 'other':6}
    ret_data = [0]*7
    for qid in qids:
        ret_data[idx_dict[qid2qtype[qid]]] += 1
    total = sum(ret_data)
    ret_data = [i/total for i in ret_data]
    average = list(average.values())
    #print(average)
    ret_data = [ 100*(ret_data[i]-average[i])/average[i] for i in range(7) ]# offset each one from the average
    maxxy = max(ret_data)
    minny = min(ret_data)
    if(maxxy > maxx):
        maxx = maxxy
    if(minny < minn):
        minn = minny
    return(ret_data)

    
if __name__ == "__main__":

    # Load correct qid matricies
    x= load_pickle('/home/jumperkables/kable_management/data/tvqa/q_type/val_q_type_dict.pickle')
    y=x['other']
    qs=[]
    when_count = 0
    for qid, qdict in y.items():
        # if 'when' in qdict['q']:
        #     when_count+=1
        if 'When' in qdict['q'].split()[0]:
            pass
            qs.append(qdict['q'])
            # if 'when' in qdict['q']:
            #     when_count+=1
        else:
            pass

    glove_valid = load_pickle( "/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/correct_qid_matrix.pickle_valid_glove")
    bert_valid = load_pickle( "/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/correct_qid_matrix.pickle_valid_bert")

    glove_train = load_pickle( "/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/correct_qid_matrix.pickle_train_glove")
    bert_train = load_pickle( "/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/correct_qid_matrix.pickle_train_bert")
    val_qids = load_pickle("/home/jumperkables/kable_management/data/tvqa/val_dict.pickle")
    train_qids = load_pickle("/home/jumperkables/kable_management/data/tvqa/train_dict.pickle")
    
    train_qids = list(train_qids.keys())
    valid_qids = list(val_qids.keys())

    popular_vote(bert_valid, valid_qids, ['V','I','R','VI','VIR','S','SI','SVI','SVIR'], "BERT")
