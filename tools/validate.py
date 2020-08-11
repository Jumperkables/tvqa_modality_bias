__author__ = "Jumperkables"

import sys, os
sys.path.insert(1, "..")
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/tvqa_modality_bias"))
#sys.path.insert(1, os.path.expanduser("~/kable_management/mk8+-tvqa"))
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from visdom_plotter import VisdomLinePlotter
import visdom
from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs
from config import BaseOptions

############## My import
import radam
from transformers import *
##############


def validate(opt, dset, model, mode="valid"):
    dset.set_mode(opt.dset) # Change mode to training here
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=True, collate_fn=pad_collate)
    if opt.bert != None:
        bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        # Word embedding lookup GloVE
        from utils import load_pickle
        idx2word = load_pickle(opt.idx2word_path)
    #valid_qids = []
    lanecheck_dict = {}
    valid_corrects = []
    if opt.disable_streams is not None:
        for d_stream in opt.disable_streams:
            if d_stream in opt.input_streams:
                opt.input_streams.remove(d_stream)
    else:
        opt.disable_streams = []
    for batch_idx, batch in enumerate(valid_loader):
        print(round(batch_idx/len(valid_loader)*100, 2), "percent complete")
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                        device=opt.device)
        if opt.lanecheck:
            sub_out, vcpt_out, vid_out, reg_out, regtopk_out, outputs = model(*model_inputs)
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        # measure accuracy and record loss
        #valid_qids += [int(x) for x in qids]

        batch_q , _    = getattr(batch, 'q')
        batch_a0, _    = getattr(batch, 'a0')
        batch_a1, _    = getattr(batch, 'a1')
        batch_a2, _    = getattr(batch, 'a2')
        batch_a3, _    = getattr(batch, 'a3')
        batch_a4, _    = getattr(batch, 'a4')
        if 'sub' in opt.input_streams:
            batch_sub, _   = getattr(batch, 'sub')
        if 'vcpt' in opt.input_streams:
            batch_vcpt, _  = getattr(batch, 'vcpt')

        # Add the ground truth to the end of each output response, and then the predicted ID for the question after that
        if 'sub' in opt.input_streams:
            sub_out = torch.cat((sub_out.cpu().squeeze(), targets.cpu().float().unsqueeze(1), pred_ids.cpu().float().unsqueeze(1)), dim=1)
        if 'vcpt' in opt.input_streams:
            vcpt_out = torch.cat((vcpt_out.cpu().squeeze(), targets.cpu().float().unsqueeze(1), pred_ids.cpu().float().unsqueeze(1)), dim=1)
        if 'imagenet' in opt.input_streams:
            vid_out = torch.cat((vid_out.cpu().squeeze(), targets.cpu().float().unsqueeze(1), pred_ids.cpu().float().unsqueeze(1)), dim=1)
        if ('regional' in opt.input_streams) and opt.regional_topk == -1:
            reg_out = torch.cat((reg_out.cpu().squeeze(), targets.cpu().float().unsqueeze(1), pred_ids.cpu().float().unsqueeze(1)), dim=1)
        if opt.regional_topk != -1:
            regtopk_out = torch.cat((regtopk_out.cpu().squeeze(), targets.cpu().float().unsqueeze(1), pred_ids.cpu().float().unsqueeze(1)), dim=1)

        # Add them to the lanecheck dictionary
        for id_idx in range(len(qids)):
            lanecheck_dict[qids[id_idx]] = {}
            if 'sub' in opt.input_streams:
                lanecheck_dict[qids[id_idx]]['sub_out']       = sub_out[id_idx]
            if 'vcpt' in opt.input_streams:
                lanecheck_dict[qids[id_idx]]['vcpt_out']      = vcpt_out[id_idx]
            if 'imagenet' in opt.input_streams:
                lanecheck_dict[qids[id_idx]]['vid_out']       = vid_out[id_idx]
            if ('regional' in opt.input_streams) and opt.regional_topk == -1:
                lanecheck_dict[qids[id_idx]]['reg_out']       = reg_out[id_idx]
            if opt.regional_topk != -1:
                lanecheck_dict[qids[id_idx]]['regtopk_out']   = regtopk_out[id_idx]

            #Vcpt
            #decode from bert
            if opt.bert != None:
                lanecheck_dict[qids[id_idx]]['q']                = bert_tok.decode(batch_q[id_idx]) 
                lanecheck_dict[qids[id_idx]]['a0']               = bert_tok.decode(batch_a0[id_idx])  
                lanecheck_dict[qids[id_idx]]['a1']               = bert_tok.decode(batch_a1[id_idx]) 
                lanecheck_dict[qids[id_idx]]['a2']               = bert_tok.decode(batch_a2[id_idx])   
                lanecheck_dict[qids[id_idx]]['a3']               = bert_tok.decode(batch_a3[id_idx])  
                lanecheck_dict[qids[id_idx]]['a4']               = bert_tok.decode(batch_a4[id_idx]) 
                if 'sub' in opt.input_streams:
                    lanecheck_dict[qids[id_idx]]['sub']          = bert_tok.decode(batch_sub[id_idx])  
                if 'vcpt' in opt.input_streams:
                    lanecheck_dict[qids[id_idx]]['vcpt']         = bert_tok.decode(batch_vcpt[id_idx])
            else:
                # Decode from GloVE
                #idx2word
                lanecheck_dict[qids[id_idx]]['q']   = [ idx2word[int(word)] for word in batch_q[id_idx] ]
                lanecheck_dict[qids[id_idx]]['a0']  = [ idx2word[int(word)] for word in batch_a0[id_idx] ]
                lanecheck_dict[qids[id_idx]]['a1']  = [ idx2word[int(word)] for word in batch_a1[id_idx] ]
                lanecheck_dict[qids[id_idx]]['a2']  = [ idx2word[int(word)] for word in batch_a2[id_idx] ]
                lanecheck_dict[qids[id_idx]]['a3']  = [ idx2word[int(word)] for word in batch_a3[id_idx] ]
                lanecheck_dict[qids[id_idx]]['a4']  = [ idx2word[int(word)] for word in batch_a4[id_idx] ]
                if 'sub' in opt.input_streams:
                    lanecheck_dict[qids[id_idx]]['sub'] = [ idx2word[int(word)] for word in batch_sub[id_idx] ]
                if 'vcpt' in opt.input_streams:
                    lanecheck_dict[qids[id_idx]]['vcpt'] = [ idx2word[int(word)] for word in batch_vcpt[id_idx] ]
    
    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    lanecheck_dict['valid_acc'] = valid_acc
    print('valid acc', valid_acc)
    from utils import save_pickle
    save_pickle(lanecheck_dict, opt.lanecheck_path+'_'+opt.dset)
    return valid_acc


if __name__ == "__main__":
    torch.manual_seed(2018)
    opt = BaseOptions().parse()
    #writer = SummaryWriter(opt.results_dir)
    #opt.writer = writer
    plotter = VisdomLinePlotter(env_name="TVQA")
    opt.plotter = plotter
    dset = TVQADataset(opt)
    if opt.bert is None:
        opt.vocab_size = len(dset.word2idx)

    # My dynamic imports
    ####
    import importlib
    print((opt.jobname))
    print((opt.modelname))
    print((opt.device))
    import_string="model."+opt.modelname
    module=importlib.import_module(import_string)
    model = module.ABC(opt)
    load_path = str(opt.best_path)
    model.load_state_dict(torch.load(load_path))
    if opt.bert is None:
        if not opt.no_glove:
            model.load_embedding(dset.vocab_embedding)

    model.to(opt.device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(size_average=False).to(opt.device)
    #
    # if(opt.lrtype == "adam"):
    #     optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
    #                                 lr=opt.lr, weight_decay=opt.wd)
    #     scheduler = None
    # if(opt.lrtype == "radam"):
    #     optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad],
    #                                 lr=opt.lr, weight_decay=opt.wd)
    #     scheduler = None
    # if(opt.lrtype == "cyclic"):
    #     optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    #     scheduler = torch.optim.CyclicLR(optimizer)
    best_acc = 0.
    early_stopping_cnt = 0
    early_stopping_flag = False
    valid_acc = validate(opt, dset, model, mode="valid")
    print(valid_acc)
    #opt.plotter.plot("accuracy", "val", opt.jobname, 0, valid_acc)
