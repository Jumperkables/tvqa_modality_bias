__author__ = 'Jumperkables'
import sys, os
sys.path.insert(1, "..")
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/tvqa_modality_bias"))
from utils import load_pickle, load_json, files_exist
import argparse
import getpass


def off_streams(filey, args):
    del filey['acc']
    correct = 0
    total = 0

    # Initialise all streams
    vcpt_flag = False
    sub_flag = False
    imagenet_flag = False
    regtopk_flag = False

    # Turn on all relevant streams
    if 'vcpt' in args.input_streams:
        vcpt_flag = True
    if 'sub' in args.input_streams:
        sub_flag = True
    if 'imagenet' in args.input_streams:
        imagenet_flag = True
    if ('regional' in args.input_streams) and (args.regional_topk != -1):
        regtopk_flag = True

    # Turn off all disabled streams
    if 'vcpt' in args.off_streams:
        vcpt_flag = False
    if 'sub' in args.off_streams:
        sub_flag = False
    if 'imagenet' in args.off_streams:
        imagenet_flag = False
    if ('regional' in args.off_streams) and (args.regional_topk != -1):
        regtopk_flag = False

    # Aggregate all responses
    for q_dict in filey.items():
        answers = [0]*5
        ground_truth = q_dict[1]['vcpt_out'][5]
        if vcpt_flag:
            answers[0] += q_dict[1]['vcpt_out'][0]
            answers[1] += q_dict[1]['vcpt_out'][1]
            answers[2] += q_dict[1]['vcpt_out'][2]
            answers[3] += q_dict[1]['vcpt_out'][3]
            answers[4] += q_dict[1]['vcpt_out'][4]
        if sub_flag:
            answers[0] += q_dict[1]['sub_out'][0]
            answers[1] += q_dict[1]['sub_out'][1]
            answers[2] += q_dict[1]['sub_out'][2]
            answers[3] += q_dict[1]['sub_out'][3]
            answers[4] += q_dict[1]['sub_out'][4]
        if imagenet_flag:
            answers[0] += q_dict[1]['vid_out'][0]
            answers[1] += q_dict[1]['vid_out'][1]
            answers[2] += q_dict[1]['vid_out'][2]
            answers[3] += q_dict[1]['vid_out'][3]
            answers[4] += q_dict[1]['vid_out'][4]
        if regtopk_flag:
            answers[0] += q_dict[1]['regtopk_out'][0]
            answers[1] += q_dict[1]['regtopk_out'][1]
            answers[2] += q_dict[1]['regtopk_out'][2]
            answers[3] += q_dict[1]['regtopk_out'][3]
            answers[4] += q_dict[1]['regtopk_out'][4]
        
        # The predicted answer from all wanted lanes
        guess = answers.index(max(answers))
        if guess == ground_truth:
            correct += 1
        total +=1
    print(correct*100/total)
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create a question type dictionary, or plot it')
    parser.add_argument('--l_path',  type=str,
                        default=None,
                        help='Where to get the lanecheck dictionary from')
    parser.add_argument("--input_streams", type=str, nargs="+", choices=["vcpt", "sub", "imagenet", "regional"], #added regional support here
                        help="input streams for the model")
    parser.add_argument("--off_streams", type=str, nargs="+", default=None, choices=["vcpt", "sub", "imagenet", "regional"], #added regional support here
                        help="turn off streams for model")
    parser.add_argument('--regional_topk',  type=int,
                        default=-1,
                        help='topk amount for regional features')

    args = parser.parse_args()

    if args.off_streams is not None:
        filey = load_pickle(args.l_path)
        off_streams(filey, args)     
    else:
        filey = load_pickle(args.l_path)
        print(filey['acc'])
