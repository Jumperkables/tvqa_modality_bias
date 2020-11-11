import sys, os
#sys.path.insert(1, os.path.expanduser("~/kable_management/mk8+-tvqa"))
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/tvqa_modality_bias"))
sys.path.insert(1, "..")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import random
#from config import BaseOptions
import utils
import numpy as np


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None
        
    def initialize(self):
        self.parser.add_argument("--lanecheck_path", type=str, 
            default=os.path.expanduser("~/kable_management/mk8+-tvqa/dataset_paper/jerry/results/tvqa_abc_svir_bert/lanecheck_dict.pickle_valid"), 
            help="Validation lane check path")

    def parse(self):
        """parse cmd line arguments and do some preprocessing"""
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        self.opt = opt
        return opt

def confusion_matrix_tn_fn(a_idx, ground_truth, prediciton):
    # if(a_idx == ground_truth) and (a_idx == prediciton):
    #     return('True Positive')
    # if(a_idx != ground_truth) and (a_idx == prediciton):
    #     return('False Positive')
    if(a_idx == ground_truth) and (a_idx != prediciton):
        return('False Negative')
    if(a_idx != ground_truth) and (a_idx != prediciton):
        return('True Negative')
    return('Ignore')

def confusion_matrix_tp_fp(a_idx, ground_truth, prediciton):
    if(a_idx == ground_truth) and (a_idx == prediciton):
        return('True Positive')
    if(a_idx != ground_truth) and (a_idx == prediciton):
        return('False Positive')
    # if(a_idx == ground_truth) and (a_idx != prediciton):
    #     return('False Negative')
    # if(a_idx != ground_truth) and (a_idx != prediciton):
    #     return('True Negative')
    return('Ignore')    




def one_plot(opt):
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    # Font settings for plot
    import matplotlib
    # matplotlib.rc('font', family='sans-serif') 
    # matplotlib.rc('font', serif='Helvetica Neue') 
    # matplotlib.rc('text', usetex='false') 
    # matplotlib.rcParams['font.family'] = 'cursive'

    # Load dictionary
    lanecheck_dict = utils.load_pickle(opt.lanecheck_path)

    # Lanecheck out
    sub_out     = []
    vcpt_out    = []
    vid_out     = []
    reg_out     = []
    regtopk_out = []

    # Check what out features are needed
    sub_flag    = True
    vcpt_flag   = True
    vid_flag    = True
    reg_flag    = True
    regtopk_flag= True
    check = random.choice(list(lanecheck_dict.values()))
    if check.get('sub_out') is None:
        sub_flag = False
    if check.get('vcpt_out') is None:
        vcpt_flag = False
    if check.get('vid_out') is None:
        vid_flag = False
    if check.get('reg_out') is None:
        reg_flag = False
    if check.get('regtopk_out') is None:
        regtopk_flag = False    

    # Iterate through the lanecheck items
    del lanecheck_dict['acc']
    for qid, q_dict in lanecheck_dict.items():
        if sub_flag:
            sub_out.append( q_dict['sub_out'] )
        if vcpt_flag:
            vcpt_out.append( q_dict['vcpt_out'] )
        if vid_flag:    
            vid_out.append( q_dict['vid_out'] )
        if reg_flag:    
            reg_out.append( q_dict['reg_out'] )
        if regtopk_flag:
            regtopk_out.append( q_dict['regtopk_out'] )
    if sub_flag:
        sub_out = np.stack(sub_out)
    if vcpt_flag:
        vcpt_out = np.stack(vcpt_out)
    if vid_flag: 
        vid_out = np.stack(vid_out)
    if reg_flag:
        reg_out = np.stack(reg_out)
    if regtopk_flag:
        regtopk_out = np.stack(regtopk_out)

    import pandas as pd

    # Plot settings
    pal_tp_fp = {"True Positive":sns.light_palette("green")[1], "False Positive":sns.light_palette("red")[1]}
    pal_tn_fn = {"True Negative":sns.light_palette("red")[1], "False Negative":sns.light_palette("orange")[1]}
    plot_no = 1

    sns.set(font_scale=3.0)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    x_labels = []
    if sub_flag:
        sub_out = [ ('Subtitles', value, aa[5], aa[6], confusion_matrix_tn_fn(a_idx, aa[5], aa[6])) for aa in sub_out for a_idx, value in enumerate(aa[:5])  ]
        sub_out = [ element for element in sub_out if element[4] != 'Ignore' ]
        x_labels.append('Subtitles')
    if vcpt_flag:
        vcpt_out = [ ('Visual Concepts', value, aa[5], aa[6], confusion_matrix_tn_fn(a_idx, aa[5], aa[6])) for aa in vcpt_out for a_idx, value in enumerate(aa[:5])  ]
        vcpt_out = [ element for element in vcpt_out if element[4] != 'Ignore' ]
        x_labels.append('Visual Concepts')
    if vid_flag:
        vid_out = [ ('ImageNet', value, aa[5], aa[6], confusion_matrix_tn_fn(a_idx, aa[5], aa[6])) for aa in vid_out for a_idx, value in enumerate(aa[:5])  ]
        vid_out = [ element for element in vid_out if element[4] != 'Ignore' ]
        x_labels.append('ImageNet')
    if regtopk_flag:
        regtopk_out = [ ('Regional Features', value, aa[5], aa[6], confusion_matrix_tn_fn(a_idx, aa[5], aa[6])) for aa in regtopk_out for a_idx, value in enumerate(aa[:5])  ]
        regtopk_out = [ element for element in regtopk_out if element[4] != 'Ignore' ]
        x_labels.append('Regional Features')
    x_labels.append('Nothing inparticular')
    #plt.xticks([])
    data = []
    data += [('', 38, 1, 1, "True Negative")]
    data += [('1', -7, 1, 1, "True Negative")]
    data += sub_out
    data += vcpt_out
    data += vid_out
    data += regtopk_out

    maxx = 0
    minn = 0
    for dtuple in data:
        if maxx < dtuple[1]:
            maxx = dtuple[1]
        if minn > dtuple[1]:
            minn = dtuple[1]
    print(maxx)
    print(minn)

    # data += [('', 38.594997, 1, 1, "False Positive")]
    #data += [('1', -5.7718792, 1, 1, "False Positive")]
    data = pd.DataFrame(data, columns=['', 'Vote Contribution', 'ground_truth', 'prediction', 'Answer Type'])
    sns.violinplot(data=data, palette=pal_tn_fn, inner="quart", linewidth=2.5, hue='Answer Type', x='', y='Vote Contribution', split=True, legend=False, legend_out=True)
    plt.title('SVIR Trained Model')
    plt.show()
    


if __name__ == "__main__":

    opt = BaseOptions().parse()
    one_plot(opt)
