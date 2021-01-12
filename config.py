__author__ = "Jumperkables"
# Adapted from code by Jie Lei

import os
import time
import torch
import argparse
import shutil
from utils import mkdirp, load_json, save_json, save_json_pretty


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.parser.add_argument("--debug", action="store_true", help="debug mode, break all loops")
        self.parser.add_argument("--results_dir_base", type=str, default=os.path.expanduser("~/kable_management/.results/test"))
        self.parser.add_argument("--log_freq", type=int, default=400, help="print, save training info")
        self.parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=3, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=100, help="mini-batch size for testing")
        self.parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--word_count_threshold", type=int, default=2, help="word vocabulary threshold")
        # model config
        self.parser.add_argument("--no_glove", action="store_true", help="not use glove vectors")
        self.parser.add_argument("--no_ts", action="store_true", help="no timestep annotation, use full length feature")
        self.parser.add_argument("--input_streams", type=str, nargs="+", choices=["vcpt", "sub", "imagenet", "regional"], #added regional support here
                                 help="input streams for the model")

        ################ Jumperkables's Additions and alterations
        self.parser.add_argument("--jobname", type=str, default="default_job_name", help="name of the job")
        self.parser.add_argument("--wandb", action="store_true", help="Plot using wandb")
        self.parser.add_argument("--modelname", type=str, default="tvqa_abc", help="name of the model ot use")
        self.parser.add_argument("--lrtype", type=str, choices=["adam", "cyclic", "radam", "lrelu"], default="adam", help="Kind of learning rate")
        self.parser.add_argument("--poolnonlin", type=str, choices=["tanh", "relu", "sigmoid", "None", "lrelu"], default="None", help="add nonlinearities to pooling layer")
        self.parser.add_argument("--pool_dropout", type=float, default=0.0, help="Dropout value for the projections")
        self.parser.add_argument("--testrun", type=bool, default=False, help="set True to stop writing and visdom")
        self.parser.add_argument("--topk", type=int, default=1, help="To use instead of max pooling")
        self.parser.add_argument("--nosub", type=bool, default=False, help="Ignore the sub stream")
        self.parser.add_argument("--noimg", type=bool, default=False, help="Ignore the imgnet stream")
        self.parser.add_argument("--noqs", action="store_true", help="dont use questions, only answers")
        self.parser.add_argument("--pool_type", type=str, default="default", choices=["default", "LinearSum", "ConcatMLP", "MCB", "MFH", "MFB", "MLB", "Block", "Tucker", "BlockTucker", "Mutan"], help="Which pooling technique to use")
        self.parser.add_argument("--pool_in_dims", type=int, nargs='+', default=[300,300], help="Input dimensions to pooling layers")
        self.parser.add_argument("--pool_out_dim", type=int, default=600, help="Output dimension to pooling layers")
        self.parser.add_argument("--pool_hidden_dim", type=int, default=1500, help="Some pooling types come with a hidden internal dimension")
        self.parser.add_argument("--bert", type=str, choices=["default", "mine", "multi_choice", "qa", "lxmert", "lxmertqa"], default=None, help="What kind of BERT model to use")
        self.parser.add_argument("--reg_feat_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/regional_features/100p.h5"),
                                    help="regional features")
        self.parser.add_argument("--my_vcpt", type=bool, default=False, help="Use my extracted visual concepts")
        self.parser.add_argument("--regional_topk", type=int, default=-1, help="Pick top-k scoring regional features across all frames")
        self.parser.add_argument("--lanecheck", type=bool, default=False, help="Validation lane checks")
        self.parser.add_argument("--lanecheck_path", type=str, help="Validation lane check path")
        self.parser.add_argument("--best_path", type=str, help="Path to best model")
        self.parser.add_argument("--disable_streams", type=str, default=None, nargs="+", choices=["vcpt", "sub", "imagenet", "regional"], #added regional support here
                                 help="disable the input stream from voting in the softmax of model outputs")
        self.parser.add_argument("--dset", choices=["valid", "test", "train"], default="valid", type=str, help="The dataset to use")
        self.parser.add_argument("--unfreeze", type=str, choices=["all", "heads", "none"], default="none", help="Fix the first few parameters of the transformer")
        ########################

        self.parser.add_argument("--n_layers_cls", type=int, default=1, help="number of layers in classifier")
        self.parser.add_argument("--hsz1", type=int, default=150, help="hidden size for the first lstm")
        self.parser.add_argument("--hsz2", type=int, default=300, help="hidden size for the second lstm")
        self.parser.add_argument("--embedding_size", type=int, default=300, help="word embedding dim")
        self.parser.add_argument("--max_sub_l", type=int, default=300, help="max length for subtitle")
        self.parser.add_argument("--max_vcpt_l", type=int, default=300, help="max length for visual concepts")
        self.parser.add_argument("--max_vid_l", type=int, default=480, help="max length for video feature")
        self.parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
        self.parser.add_argument("--no_normalize_v", action="store_true", help="do not normalize video featrue")
        # Data paths
        self.parser.add_argument("--train_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/tvqa_train_processed.json"),
                                 help="train set path")
        self.parser.add_argument("--valid_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/tvqa_val_processed.json"),
                                 help="valid set path")
        self.parser.add_argument("--test_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/tvqa_test_public_processed.json"),
                                 help="test set path")
        self.parser.add_argument("--glove_path", type=str, default=os.path.expanduser("~/kable_management/data/word_embeddings/glove.6B.300d.txt"),
                                 help="GloVe pretrained vector path")
        self.parser.add_argument("--vcpt_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/vcpt_features/det_visual_concepts_hq.pickle"),
                                 help="visual concepts feature path")
        self.parser.add_argument("--vid_feat_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/imagenet_features/tvqa_imagenet_pool5_hq.h5"),
                                 help="imagenet feature path")
        self.parser.add_argument("--vid_feat_size", type=int, default=2048,
                                 help="visual feature dimension")
        self.parser.add_argument("--word2idx_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/cache/word2idx.pickle"),
                                 help="word2idx cache path")
        self.parser.add_argument("--idx2word_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/cache/idx2word.pickle"),
                                 help="idx2word cache path")
        self.parser.add_argument("--vocab_embedding_path", type=str, default=os.path.expanduser("~/kable_management/data/tvqa/cache/vocab_embedding.pickle"),
                                 help="vocab_embedding cache path")
        self.parser.add_argument("--deep_cca", action="store_true", help="To perform deep cca")
        self.parser.add_argument("--deep_cca_layers", type=int, default=2, help="How many layers in deep cca")
        self.parser.add_argument("--dual_stream", action="store_true", help="Activate the dual stream lane that will use BLP")
        self.parser.add_argument("--rubi", action="store_true", help="implement RUBI protocol")
        self.parser.add_argument("--rubi_qloss_weight", type=float, default=1.0, help="question loss weight from RUBiCriterion")

        self.initialized = True

    def display_save(self, options, results_dir):
        """save config info for future reference, and print"""
        args = vars(options)  # type == dict
        # Display settings
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print(('%s: %s' % (str(k), str(v))))
        print('-------------- End ----------------')

        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(results_dir, 'opt.json')  # not yaml file indeed
            save_json_pretty(args, option_file_path)

    def parse(self):
        """parse cmd line arguments and do some preprocessing"""
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        results_dir = opt.results_dir_base #+ time.strftime("_%Y_%m_%d_%H_%M_%S")

        if isinstance(self, TestOptions) and False:
            options = load_json(os.path.join("results", opt.model_dir, "opt.json"))
            for arg in options:
                setattr(opt, arg, options[arg])
        else:
            if(os.path.isdir(results_dir)):
                if not opt.lanecheck:
                    shutil.rmtree(results_dir)
                    os.makedirs(results_dir)
            else:
                os.makedirs(results_dir)
            self.display_save(opt, results_dir)

        opt.normalize_v = not opt.no_normalize_v
        opt.device = torch.device("cuda:%d" % opt.device if opt.device >= 0 else "cpu")
        opt.with_ts = not opt.no_ts
        opt.input_streams = [] if opt.input_streams is None else opt.input_streams
        opt.vid_feat_flag = True if "imagenet" in opt.input_streams else False
        opt.h5driver = None if opt.no_core_driver else "core"
        opt.results_dir = results_dir

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--model_dir", type=str, help="dir contains the model file")
        self.parser.add_argument("--mode", type=str, default="valid", help="valid/test")


if __name__ == "__main__":
    import sys
    sys.argv[1:] = ["--input_streams", "vcpt"]
    opt = BaseOptions().parse()

