__author__ = 'Jumperkables'
import sys, os
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/tvqa_modality_bias"))
#sys.path.insert(1, os.path.expanduser("~/kable_management/mk8+-tvqa"))
sys.path.insert(1, "..")
from utils import load_pickle, save_pickle, load_json, files_exist
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns


# Create lists of eac different question type as per TVQA guidelines
## i.e. classify the question by it's first word and then 
def create_q_type_dict(dset, name):
    what_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}   # Potentially do multiple for if multiple of these clauses are used, for now just take first
    who_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    why_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    where_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    how_dict = {'after': {}, 'when': {}, 'before':{}, 'other':{}}
    which_dict = {}
    other_dict = {} # Anything that fails to classify by first word goes here

    # Iterate over questions and check, qid:question in each type
    for idx, question_dict in enumerate(dset):
        question = [ word.lower() for word in question_dict['q'].split() ]
        if question[0] == 'what':
            if 'when' in question:
                what_dict['when'][question_dict['qid']] = question_dict
            elif 'before' in question:
                what_dict['before'][question_dict['qid']] = question_dict
            elif 'after' in question:
                what_dict['after'][question_dict['qid']] = question_dict
            else:
                what_dict['other'][question_dict['qid']] = question_dict

        elif question[0] == 'who':
            if 'when' in question:
                who_dict['when'][question_dict['qid']] = question_dict
            elif 'before' in question:
                who_dict['before'][question_dict['qid']] = question_dict
            elif 'after' in question:
                who_dict['after'][question_dict['qid']] = question_dict
            else:
                who_dict['other'][question_dict['qid']] = question_dict

        elif question[0] == 'why':
            if 'when' in question:
                why_dict['when'][question_dict['qid']] = question_dict
            elif 'before' in question:
                why_dict['before'][question_dict['qid']] = question_dict
            elif 'after' in question:
                why_dict['after'][question_dict['qid']] = question_dict
            else:
                why_dict['other'][question_dict['qid']] = question_dict

        elif question[0] == 'where':
            if 'when' in question:
                where_dict['when'][question_dict['qid']] = question_dict
            elif 'before' in question:
                where_dict['before'][question_dict['qid']] = question_dict
            elif 'after' in question:
                where_dict['after'][question_dict['qid']] = question_dict
            else:
                where_dict['other'][question_dict['qid']] = question_dict
        
        elif question[0] == 'how':
            if 'when' in question:
                how_dict['when'][question_dict['qid']] = question_dict
            elif 'before' in question:
                how_dict['before'][question_dict['qid']] = question_dict
            elif 'after' in question:
                how_dict['after'][question_dict['qid']] = question_dict
            else:
                how_dict['other'][question_dict['qid']] = question_dict
        
        elif question[0] == 'which':
            which_dict[question_dict['qid']] = question_dict

        else:
            other_dict[question_dict['qid']] = question_dict

    # Create the overall question type dictionary and save
    q_type_dict = {
        'what': what_dict,
        'who': who_dict,
        'why': why_dict,
        'where': where_dict,
        'how': how_dict,
        'which': which_dict,
        'other': other_dict
    }
    #save_pickle(q_type_dict, os.path.expanduser("~/kable_management/data/tvqa/q_type/"+name+"_q_type_dict.pickle"))
    return q_type_dict

def plot_qtype_dict(qtype_dict, name):
    # Get the counts of each question type and sub-type
    what_after, what_before, what_when, what_other = len(qtype_dict['what']['after']), len(qtype_dict['what']['before']), len(qtype_dict['what']['when']), len(qtype_dict['what']['other'])
    who_after, who_before, who_when, who_other = len(qtype_dict['who']['after']), len(qtype_dict['who']['before']), len(qtype_dict['who']['when']), len(qtype_dict['who']['other'])
    why_after, why_before, why_when, why_other = len(qtype_dict['why']['after']), len(qtype_dict['why']['before']), len(qtype_dict['why']['when']), len(qtype_dict['why']['other'])
    where_after, where_before, where_when, where_other = len(qtype_dict['where']['after']), len(qtype_dict['where']['before']), len(qtype_dict['where']['when']), len(qtype_dict['where']['other'])
    how_after, how_before, how_when, how_other = len(qtype_dict['how']['after']), len(qtype_dict['how']['before']), len(qtype_dict['how']['when']), len(qtype_dict['how']['other'])

    what = what_after + what_before + what_when + what_other
    who = who_after + who_before + who_when + who_other
    why = why_after + why_before + why_when + why_other
    where = where_after + where_before + where_when + where_other
    how = how_after + how_before + how_when + how_other
    which = len(qtype_dict['which'])
    other = len(qtype_dict['other'])
    
    # Plot details
    ####### Font
    import matplotlib
    #matplotlib.rc('xtick', labelsize=20)     
    #matplotlib.rc('ytick', labelsize=20)
    # font = {
    #     'family' : 'sans-serif',
    #     'weight' : 'normal',
    #     'size'   : 18,
    #     #'fontname' : 'Arial'
    # }	
    # matplotlib.rc('font', **font)
    #import ipdb; ipdb.set_trace()
    matplotlib.rc('font', family='sans-serif') 
    matplotlib.rc('font', serif='Helvetica Neue') 
    matplotlib.rc('text', usetex='false') 
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['font.family'] = 'cursive'
    #######################################
    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.3

    # Outer ring
    #cm = plt.get_cmap("tab20b")
    #cout = cm([0, 4, 8, 12, 16, 20, 21])
    cm = plt.get_cmap("Dark2")
    counts = [what, who, why, where, how, which, other]
    cout = cm([0,1,2,3,5,7,7])
    labels=[
        'What ('+str(round(100*what/sum(counts),1))+'%)',
        'Who ('+str(round(100*who/sum(counts),1))+'%)',
        'Why ('+str(round(100*why/sum(counts),1))+'%)',
        'Where ('+str(round(100*where/sum(counts),1))+'%)',
        'How ('+str(round(100*how/sum(counts),1))+'%)',
        'Which ('+str(round(100*which/sum(counts),1))+'%)',
        'Other ('+str(round(100*other/sum(counts),1))+'%)'
    ]
    # 'Who', 'Why', 'Where', 'How', 'Which', 'Other']
    pie, _ = ax.pie(counts, radius=1, labels=labels, colors=cout)
    plt.setp( pie, width=width, edgecolor='white')

    # Inner ring
    # cm = plt.get_cmap("tab20b")
    # cin = cm([i for i in range(22)])
    cm = plt.get_cmap("tab20c")
    cin = cm([17, 18, 19,0]*5+[16, 16])
    counts = [what_after, what_before, what_when, what_other, who_after, who_before, who_when, who_other, why_after, why_before, why_when, why_other, where_after, where_before, where_when, where_other, how_after, how_before, how_when, how_other, which, other]
    #labels = ['what_after', 'what_before', 'what_when', 'what_other', 'who_after', 'who_before', 'who_when', 'who_other', 'why_after', 'why_before', 'why_when', 'why_other', 'where_after', 'where_before', 'where_when', 'where_other', 'how_after', 'how_before', 'how_when', 'how_other', 'which', 'other']
    #labels = ['after', 'before', 'when', 'other', 'after', 'before', 'when', 'other', 'after', 'before', 'when', 'other', 'after', 'before', 'when', 'other', 'after', 'before', 'when', 'other', 'which', 'other']
    #labels = ['after', 'before', 'when', '', 'after', 'before', 'when', '', 'after', 'before', 'when', '', 'after', 'before', 'when', '', 'after', 'before', 'when', '', '', '']
    #labels = ['A', 'B', 'W', '']*5+['','']
    pie2, _ = ax.pie(counts, radius=1-width, labeldistance=0.7, colors=cin)#labels=labels, labeldistance=0.7, colors=cin)
    plt.setp( pie2, width=width, edgecolor='white')
    plt.title(name+' Dataset Question Type Distribution')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=cm(17), lw=4),
                    Line2D([0], [0], color=cm(18), lw=4),
                    Line2D([0], [0], color=cm(19), lw=4)]
    # import ipdb; ipdb.set_trace()
    ax.legend(custom_lines,['When','Before','After'])
    plt.show()
    print(name, "Plotted")



def plot_acc_by_type(args, model_name):
    val_qtype_dict = load_pickle(os.path.expanduser("~/kable_management/data/tvqa/q_type/val_q_type_dict.pickle"))
    lanecheck_dict = load_pickle(args.lanecheck_path)
    del lanecheck_dict['valid_acc'] # Validation accuracy kept here for other code, delete it to avoid problems
    correct_dict = {}   # Dictionary for each question in the validation set storing if the model got that question right
    check_dict = random.choice(list(lanecheck_dict.values()))
    for key in check_dict.keys():    # Any of the possible feature lanes the model may have
        if key in ['sub_out', 'vcpt_out', 'vid_out', 'reg_out', 'regtopk_out']:
            check_key = key
            break  
    for qid, question_dict in lanecheck_dict.items():    
        correct = (question_dict[check_key][5] == question_dict[check_key][6])  # 5 is ground truth, 6 is predicted
        correct_dict[qid] = correct
    
    what_correct, what_not = 0, 0
    who_correct, who_not = 0, 0
    why_correct, why_not = 0, 0
    where_correct, where_not = 0, 0
    how_correct, how_not = 0, 0
    which_correct, which_not = 0, 0
    other_correct, other_not = 0, 0
    for sub_type_dict in val_qtype_dict['what'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                what_correct += 1
            else:
                what_not += 1
    for sub_type_dict in val_qtype_dict['who'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                who_correct += 1
            else:
                who_not += 1
    for sub_type_dict in val_qtype_dict['why'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                why_correct += 1
            else:
                why_not += 1
    for sub_type_dict in val_qtype_dict['where'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                where_correct += 1
            else:
                where_not += 1
    for sub_type_dict in val_qtype_dict['how'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                how_correct += 1
            else:
                how_not += 1
    for qid in val_qtype_dict['which'].keys():
        if correct_dict[qid]:
            which_correct += 1
        else:
            which_not += 1
    for qid in val_qtype_dict['other'].keys():
        if correct_dict[qid]:
            other_correct += 1
        else:
            other_not += 1
    what = what_correct+what_not
    who = who_correct+who_not
    why = why_correct+why_not
    where = where_correct+where_not
    how = how_correct+how_not
    which = which_correct+which_not
    other = other_correct+other_not
    
    # Plot details
    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.3
    cm = plt.get_cmap("tab20c")

    # Outer ring
    cout = cm([0, 4, 8, 12, 16, 20, 21])
    counts = [what, who, why, where, how, which, other]
    labels=['what', 'who', 'why', 'where', 'how', 'which', 'other']
    pie, _ = ax.pie(counts, radius=1, labels=labels, colors=cout)
    plt.setp( pie, width=width, edgecolor='white')

    # Inner Ring
    cin = cm([1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 23, 24])
    counts = [what_correct, what_not, who_correct, who_not, why_correct, why_not, where_correct, where_not, how_correct, how_not, which_correct, which_not, other_correct, other_not]
    labels = ['correct\n'+str(round(what_correct/(what_correct+what_not),4)*100)+'%', 'not\n'+str(round(what_not/(what_correct+what_not),4)*100)+'%', 
        'correct\n'+str(round(who_correct/(who_correct+who_not),4)*100)+'%', 'not\n'+str(round(who_not/(who_correct+who_not),4)*100)+'%',
        'correct\n'+str(round(why_correct/(why_correct+why_not),4)*100)+'%', 'not\n'+str(round(why_not/(why_correct+why_not),4)*100)+'%',
        'correct\n'+str(round(where_correct/(where_correct+where_not),4)*100)+'%', 'not\n'+str(round(where_not/(where_correct+where_not),4)*100)+'%',
        'correct\n'+str(round(how_correct/(how_correct+how_not),4)*100)+'%', 'not\n'+str(round(how_not/(how_correct+how_not),4)*100)+'%',
        'correct\n'+str(round(which_correct/(which_correct+which_not),4)*100)+'%', 'not\n'+str(round(which_not/(which_correct+which_not),4)*100)+'%',
        'correct\n'+str(round(other_correct/(other_correct+other_not),4)*100)+'%', 'not\n'+str(round(other_not/(other_correct+other_not),4)*100)+'%',]
    pie2, _ = ax.pie(counts, radius=1-width, labels=labels, labeldistance=0.7, colors=cin)
    plt.setp( pie2, width=width, edgecolor='white')
    total_acc = round( (what_correct+who_correct+why_correct+where_correct+how_correct+which_correct+other_correct)/(what+who+why+where+how+which+other) ,4)*100
    plt.title(model_name+' Dataset Question Type Distribution.\n Overall Acc:'+str(total_acc))
    plt.show()
    print(model_name, "Plotted")





def qtype_dist(lanecheck_path):
    lanecheck_dict = load_pickle(lanecheck_path)
    val_qtype_dict = load_pickle(os.path.expanduser("~/kable_management/data/tvqa/q_type/val_q_type_dict.pickle"))
    acc = lanecheck_dict['acc']
    del lanecheck_dict['acc'] # Validation accuracy kept here for other code, delete it to avoid problems
    correct_dict = {}   # Dictionary for each question in the validation set storing if the model got that question right
    check_dict = random.choice(list(lanecheck_dict.values()))
    for key in check_dict.keys():    # Any of the possible feature lanes the model may have
        if key in ['sub_out', 'vcpt_out', 'vid_out', 'reg_out', 'regtopk_out']:
            check_key = key
            break  
    for qid, question_dict in lanecheck_dict.items():    
        correct = (question_dict[check_key][5] == question_dict[check_key][6])  # 5 is ground truth, 6 is predicted
        correct_dict[qid] = correct

    what_correct, what_not = 0, 0
    who_correct, who_not = 0, 0
    why_correct, why_not = 0, 0
    where_correct, where_not = 0, 0
    how_correct, how_not = 0, 0
    which_correct, which_not = 0, 0
    other_correct, other_not = 0, 0
    for sub_type_dict in val_qtype_dict['what'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                what_correct += 1
            else:
                what_not += 1
    for sub_type_dict in val_qtype_dict['who'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                who_correct += 1
            else:
                who_not += 1
    for sub_type_dict in val_qtype_dict['why'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                why_correct += 1
            else:
                why_not += 1
    for sub_type_dict in val_qtype_dict['where'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                where_correct += 1
            else:
                where_not += 1
    for sub_type_dict in val_qtype_dict['how'].values():
        for qid in sub_type_dict:
            if correct_dict[qid]:
                how_correct += 1
            else:
                how_not += 1
    for qid in val_qtype_dict['which'].keys():
        if correct_dict[qid]:
            which_correct += 1
        else:
            which_not += 1
    for qid in val_qtype_dict['other'].keys():
        if correct_dict[qid]:
            other_correct += 1
        else:
            other_not += 1
    what = what_correct+what_not
    who = who_correct+who_not
    why = why_correct+why_not
    where = where_correct+where_not
    how = how_correct+how_not
    which = which_correct+which_not
    other = other_correct+other_not
    ret_data = [
        what_correct, what_not, 
        who_correct, who_not, 
        why_correct, why_not, 
        where_correct, where_not, 
        how_correct, how_not,
        which_correct, which_not, 
        other_correct, other_not
    ]
    return ret_data, acc





def magnum_opus():
    #jerry_subpath = '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/jerry/results/'
    #ncc_subpath = '/home/jumperkables/kable_management/mk8+-tvqa/dataset_paper/ncc/results/'
    #paths = [
    #     ncc_subpath+'tvqa_abc_v/',
    #     ncc_subpath+'tvqa_abc_v_bert/',
    #     ncc_subpath+'tvqa_abc_i/',
    #     ncc_subpath+'tvqa_abc_i_bert/',
    #     jerry_subpath+'tvqa_abc_r/',
    #     jerry_subpath+'tvqa_abc_r_bert/',
    #     ncc_subpath+'tvqa_abc_vi/',
    #     ncc_subpath+'tvqa_abc_vi_bert/',
    #     jerry_subpath+'tvqa_abc_vir/',
    #     jerry_subpath+'tvqa_abc_vir_bert/',
    #     ncc_subpath+'tvqa_abc_s/',
    #     ncc_subpath+'tvqa_abc_s_bert/',
    #     ncc_subpath+'tvqa_abc_si/',
    #     ncc_subpath+'tvqa_abc_si_bert/',
    #     ncc_subpath+'tvqa_abc_svi/',
    #     ncc_subpath+'tvqa_abc_svi_bert/',
    #     jerry_subpath+'tvqa_abc_svir/',
    #     jerry_subpath+'tvqa_abc_svir_bert/'
    # ]
     train_plot_paths = [path+'lanecheck_dict.pickle_train' for path in paths]
     valid_plot_paths = [path+'lanecheck_dict.pickle_valid' for path in paths]
    # Here put a list of paths to lanecheck dictionaries for training or validation outputs.i
    # With corresponding labels for each lanecheck model
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

    # An example breakdown of question types correct vs incorrect
    qtype_dict = load_pickle('/home/jumperkables/kable_management/data/tvqa/q_type/val_q_type_dict.pickle')
    category_names = [
        'what_correct', 'what_not',
        'who_correct', 'who_not',
        'why_correct', 'why_not',
        'where_correct', 'where_not',
        'how_correct', 'how_not',
        'which_correct', 'which_not',
        'other_correct', 'other_not'
    ]
    
    glove_labels = [labels[i] for i in range(len(labels)) if i%2 == 0 ]
    bert_labels = [labels[i] for i in range(len(labels)) if i%2 == 1 ]

    glove_paths = [valid_plot_paths[i] for i in range(len(valid_plot_paths)) if i%2 == 0 ]
    bert_paths = [valid_plot_paths[i] for i in range(len(valid_plot_paths)) if i%2 == 1 ]

    # GloVe
    glove_results = {}
    glove_accuracies = {}
    for i, path in enumerate(glove_paths):
        model_results, model_acc = qtype_dist(path) # Model name: array of stacked data, accuracy at the end
        glove_results[glove_labels[i]] = model_results
        glove_accuracies[glove_labels[i]] = model_acc

    # Bert
    bert_results = {}
    bert_accuracies = {}
    for i, path in enumerate(bert_paths):
        model_results, model_acc = qtype_dist(path) # Model name: array of stacked data, accuracy at the end
        bert_results[bert_labels[i]] = model_results
        bert_accuracies[bert_labels[i]] = model_acc


    #Glove
    survey(glove_results, category_names, glove_accuracies)
    plt.show()
    # BERT
    # survey(bert_results, category_names, bert_accuracies)
    # plt.show()


def survey(results, category_names, accuracies):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    # Collecting data for plots
    labels = list(results.keys())
    accuracies = list(accuracies.values())
    accuracies = [100*i for i in accuracies]
    data = np.array(list(results.values()))
    
    # Colours details
    #correct_colours = plt.get_cmap('tab20')([1,3,5,7,9,11,13])
    correct_colours = plt.get_cmap('Dark2')([0,1,2,3,5,6,8])
    #incorrect_colours = plt.get_cmap('tab20')([0,2,4,6,8,10,12])
    
    # Get the end 
    correct_end = np.ndarray((7, len(data)))
    for i in range(len(data)):
        for j in range(0, 7):
            correct_end[j][i] = data[i][2*j]*100/(data[i][2*j]+data[i][2*j+1])

    # Figure settings
    import matplotlib
    plt.rc('font', size=24)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(data))*2
    width = 0.2       # the width of the bars
    width1 = width *1

    #Set font size
    ax.set_xlabel('',fontsize = 24) #xlabel
    ax.set_ylabel('%', fontsize = 24)#ylabel
    ax.set_xticks(ind)
    ax.set_xticklabels( labels,  fontsize = 24 )

    # #Plot incorect bars
    # rects_iwhat = ax.bar(ind-3*width, [100]*len(data), width1, color=incorrect_colours[0])
    # rects_iwho  = ax.bar(ind-2*width, [100]*len(data), width1, color=incorrect_colours[1])
    # rects_iwhy  = ax.bar(ind-1*width, [100]*len(data), width1, color=incorrect_colours[2])
    # rects_iwhere= ax.bar(ind, [100]*len(data), width1, color=incorrect_colours[3])
    # rects_ihow  = ax.bar(ind+1*width, [100]*len(data), width1, color=incorrect_colours[4])
    # rects_iwhich= ax.bar(ind+2*width, [100]*len(data), width1, color=incorrect_colours[5])
    # rects_iother= ax.bar(ind+3*width, [100]*len(data), width1, color=incorrect_colours[6])

    # Plot correct
    rects_cwhat = ax.bar(ind-3*width, correct_end[0], width1, color=correct_colours[0], edgecolor='black')
    rects_cwho  = ax.bar(ind-2*width, correct_end[1], width1, color=correct_colours[1], edgecolor='black')
    rects_cwhy  = ax.bar(ind-1*width, correct_end[2], width1, color=correct_colours[2], edgecolor='black')
    rects_cwhere= ax.bar(ind, correct_end[3], width1, color=correct_colours[3], edgecolor='black')
    rects_chow  = ax.bar(ind+1*width, correct_end[4], width1, color=correct_colours[4], edgecolor='black')
    rects_cwhich= ax.bar(ind+2*width, correct_end[5], width1, color=correct_colours[5], edgecolor='black')
    rects_cother= ax.bar(ind+3*width, correct_end[6], width1, color=correct_colours[6], edgecolor='black')

    #Plot Accuracies
    acc_lines = ax.hlines(accuracies, ind-3.5*width, ind+3.5*width, linestyle='dashed', color='r')
    
    # Legend and appropriate handles
    ax.legend(loc='upper left')#, fontsize='small')
    


    return fig, ax


# argparse options to run this to extract question types
# or to plot onto pie charts
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create a question type dictionary, or plot it')
    parser.add_argument('--action',  type=str,
                        default=None, choices=['create_dict', 'plot', 'acc_by_type_plot', 'magnum_opus'],
                        help='create_qtype_dict or plot')
    parser.add_argument('--lanecheck_path',  type=str,
                        default=None,
                        help='Where to get the lanecheck dictionary from')
    parser.add_argument('--model',  type=str,
                        default=None,
                        help='Model name')
    args = parser.parse_args()
    if not args.action:
        sys.exit()

    # If you want to create a dictinary
    if args.action == 'create_dict':    
        train_dset = load_json(os.path.expanduser("~/kable_management/data/tvqa/tvqa_train_processed.json"))
        val_dset = load_json(os.path.expanduser("~/kable_management/data/tvqa/tvqa_val_processed.json"))
        test_dset = load_json(os.path.expanduser("~/kable_management/data/tvqa/tvqa_test_public_processed.json"))
        total_dset = []
        total_dset += train_dset
        total_dset += val_dset
        total_dset += test_dset

        create_q_type_dict(train_dset, "train")
        create_q_type_dict(val_dset, "val")
        create_q_type_dict(test_dset, "test")
        create_q_type_dict(total_dset, 'total')
    
    # Plot the dictionary of question types
    if args.action == 'plot':
        train_qtype_dict = load_pickle(os.path.expanduser("~/kable_management/data/tvqa/q_type/train_q_type_dict.pickle"))
        val_qtype_dict = load_pickle(os.path.expanduser("~/kable_management/data/tvqa/q_type/val_q_type_dict.pickle"))
        test_qtype_dict = load_pickle(os.path.expanduser("~/kable_management/data/tvqa/q_type/test_q_type_dict.pickle"))
        total_qtype_dict = load_pickle(os.path.expanduser("~/kable_management/data/tvqa/q_type/total_q_type_dict.pickle"))

        #plot_qtype_dict(train_qtype_dict, 'Train')
        #plot_qtype_dict(val_qtype_dict, 'Val')
        #plot_qtype_dict(test_qtype_dict, 'Test')
        plot_qtype_dict(total_qtype_dict, 'Total')
        

    # Plot the accuracy of the model by question types
    if args.action == 'acc_by_type_plot':
        plot_acc_by_type(args, args.model)
    
    # Drink the big one
    if args.action == 'magnum_opus':
        magnum_opus()
