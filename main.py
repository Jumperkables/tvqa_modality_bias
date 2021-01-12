__author__ = "Jie Lei, edited by Jumperkables"

import os, sys
import importlib
sys.path.insert(1, "./tools")
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from visdom_plotter import VisdomLinePlotter
from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs
from config import BaseOptions
from rubi_criterion import RUBiCriterion 

############## 
import radam
from transformers import *
from utils import load_pickle, save_pickle
import wandb
##############


def train(opt, dset, model, criterion, optimizer, epoch, previous_best_acc, scheduler):
    dset.set_mode("train")
    if opt.rubi: # Model and the question/subtitle only rubi style model ill be packed together, unpack here
        model, rubi_model = model
        rubi_model.train()
    model.train()
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=True, collate_fn=pad_collate)
    train_loss = []
    valid_acc_log = ["batch_idx\tacc"]
    train_corrects = []
    torch.set_grad_enabled(True)
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Process inputs
        if(opt.lrtype == "cyclic"):
            scheduler.batch_step()
        model_inputs, targets, _ = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                     device=opt.device)
        # model output
        if opt.dual_stream:
            outputs = model(*model_inputs)
            if opt.lanecheck:
                raise Exception("Not implemeneted lanechecking with dual stream")
        else:
            outputs = model(*model_inputs)
            if opt.rubi:
                rubi_outputs = rubi_model(*model_inputs)
        
        # Loss 
        if not opt.rubi:
            if opt.lanecheck:
                loss = criterion(outputs[-1], targets)
            else:
                loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            if opt.lanecheck:
                rubi_in = { # This may be confusing, but this is because of my naming scheme conflicting with Remi's
                        'logits':None,
                        'logits_q':rubi_outputs[-1],
                        'logits_rubi':outputs[-1]
                }
            else:
                rubi_in = { # This may be confusing, but this is because of my naming scheme conflicting with Remi's
                        'logits':None,
                        'logits_q':rubi_outputs,
                        'logits_rubi':outputs
                }

            rubi_targets = {
                    'class_id':targets
            }
            losses = criterion(rubi_in, rubi_targets)
            loss , q_loss = losses['loss_mm_q'], losses['loss_q'] # loss is the fused rubi loss
            optimizer.zero_grad()
            loss.backward()
            q_loss.backward()   # push the question loss backwards too
            optimizer.step()
        opt.plotter.text_plot(opt.jobname+" epoch", opt.jobname+" "+str(epoch))

        # measure accuracy and record loss
        train_loss.append(loss.item())
        if opt.lanecheck:
            pred_ids = outputs[-1].data.max(1)[1]
        else:
            pred_ids = outputs.data.max(1)[1]
        train_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()
        if ((batch_idx % opt.log_freq) == (opt.log_freq-1)):#15):
            niter = epoch * len(train_loader) + batch_idx
            train_acc = sum(train_corrects) / float(len(train_corrects))
            train_loss = sum(train_loss) / float(len(train_loss))#from train_corrects
            
            # Plotting
            if opt.testrun == False:
                opt.writer.add_scalar("Train/Acc", train_acc, niter)
                opt.writer.add_scalar("Train/Loss", train_loss, niter)
                opt.plotter.plot("accuracy", "train", opt.jobname, niter, train_acc)
                opt.plotter.plot("loss", "train", opt.jobname, niter, train_loss)

            # Validation
            if opt.dual_stream or opt.deep_cca:
                valid_acc, _ = validate(opt, dset, model, mode="valid")
            elif not opt.lanecheck:
                valid_acc, _ = validate(opt, dset, model, mode="valid")
            else:
                valid_acc, val_lanecheck_dict = validate_lanecheck(opt, dset, model, mode="valid")
            if opt.testrun == False:
                #opt.writer.add_scalar("Valid/Loss", valid_loss, niter)
                opt.plotter.plot("accuracy", "val", opt.jobname, niter, valid_acc)
                #opt.plotter.plot("loss", "val", opt.jobname, niter, valid_loss)
            valid_log_str = "%02d\t%.4f" % (batch_idx, valid_acc)
            valid_acc_log.append(valid_log_str)
            
            # If this is the best run yet
            if valid_acc > previous_best_acc:
                previous_best_acc = valid_acc

                # Plot best accuracy so far in text box
                if opt.testrun == False:
                    opt.plotter.text_plot(opt.jobname+" val", opt.jobname+" val "+str(round(previous_best_acc, 4)))

                # Save the predictions for validation and training datasets, and the state dictionary of the model
                #_, train_lanecheck_dict = validate_lanecheck(opt, dset, model, mode="train")
                if ((not opt.dual_stream) and (not opt.deep_cca)):
                    if opt.lanecheck:
                        save_pickle(val_lanecheck_dict, opt.lanecheck_path+'_valid')
                torch.save(model.state_dict(), os.path.join(opt.results_dir, "best_valid.pth"))
            # Cleaner, newer, wandb logger code    
            if opt.wandb:
                wandb.log({
                    "Val Acc"   :valid_acc,
                    "Best Acc"  :previous_best_acc,
                    "Train Loss":train_loss
                })

            # reset to train
            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")
            train_corrects = []
            train_loss = []

        if opt.debug:
            break

    # additional log
    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    return previous_best_acc







def validate_lanecheck(opt, dset, model, mode="valid"):
    dset.set_mode(mode) # Change mode to training here
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)
    lanecheck_dict = {}
    valid_corrects = []
    #opt.lanecheck = True
    if opt.disable_streams is not None:
        for d_stream in opt.disable_streams:
            if d_stream in opt.input_streams:
                opt.input_streams.remove(d_stream)
    else:
        opt.disable_streams = []

    for batch_idx, batch in enumerate(valid_loader):
        # Accuracy
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                        device=opt.device)
        sub_out, vcpt_out, vid_out, reg_out, regtopk_out, outputs = model(*model_inputs)
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        # Add the ground truth to the end of each output response, and then the predicted ID for the question after that
        if 'sub' in opt.input_streams and not opt.dual_stream:
            sub_out = torch.cat((sub_out.cpu().squeeze(), targets.cpu().float().unsqueeze(1), pred_ids.cpu().float().unsqueeze(1)), dim=1)
        if 'vcpt' in opt.input_streams and not opt.dual_stream:
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
    
    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    lanecheck_dict['acc'] = valid_acc
    #opt.lanecheck = False
    return valid_acc, lanecheck_dict






def validate(opt, dset, model, mode="valid"):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    valid_qids = []
    valid_loss = []
    valid_corrects = []
    for _, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                        device=opt.device)
        if opt.dual_stream:
            outputs = model(*model_inputs)
        else:
            outputs = model(*model_inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        valid_qids += [int(x) for x in qids]
        valid_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        if opt.debug:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_loss) / float(len(valid_loss))

    return valid_acc, valid_loss





if __name__ == "__main__":
    torch.manual_seed(2018)
    opt = BaseOptions().parse()
    writer = SummaryWriter(opt.results_dir)
    opt.writer = writer
    plotter = VisdomLinePlotter(env_name=opt.jobname)

    if opt.wandb:
        wandb.init(project="a_vs_c", name=opt.jobname)
        wandb.config.update(opt)

    opt.plotter = plotter
    dset = TVQADataset(opt)
    if opt.bert is None:
        opt.vocab_size = len(dset.word2idx)
    if opt.disable_streams is None:
        opt.disable_streams = []

    # My dynamic imports
    ####
    print((opt.jobname))
    print((opt.modelname))
    print((opt.device))
    fyle, model = opt.modelname.split(".")
    import_string="model."+fyle
    module = importlib.import_module(import_string)
    if model == "Hopfield":
        model = module.Hopfield(opt)
    elif model == "Lxmert_adapt":
        model = module.Lxmert_adapt(opt)
    else:
        raise NotImplementedError("The model import functionality is edited. You may need to make a minor change here to get the right model")

    #if True:
    #    model.load_state_dict(torch.load("/home/jumperkables/kable_management/projects/tvqa_modality_bias/.results/mk0_i_lxmert/best_valid.pth"))

    if opt.rubi:
        # My rubi implementation uses a question/subtitle only model to contrast against a question/subtitle/imagenet feature
        rubi_opt = opt
        rubi_opt.input_streams = ['sub']
        rubi_model = module.ABC(rubi_opt) # rubi_model is the qs model, regularly named model is the full si with questions model
        if opt.bert is None:
            if not opt.no_glove:
                rubi_model.load_embedding(dset.vocab_embedding)
        rubi_model.to(opt.device)
        if(opt.lrtype == "radam"):
            rubi_optimizer = radam.RAdam([p for p in rubi_model.parameters() if p.requires_grad],
                                        lr=opt.lr, weight_decay=opt.wd)
            rubi_scheduler = None
        else:
            raise Exception("Not implemented rubi with non-radam optimiser")


    if opt.bert is None: # Sorting BERT or GloVe embedding
        if not opt.no_glove:
            model.load_embedding(dset.vocab_embedding)

    model.to(opt.device)
    cudnn.benchmark = True
    if opt.rubi:
        rubi_criterion = RUBiCriterion(opt.rubi_qloss_weight).to(opt.device)
    else:
        criterion = nn.CrossEntropyLoss(size_average=False).to(opt.device)

    #
    if(opt.lrtype == "adam"):
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                    lr=opt.lr, weight_decay=opt.wd)
        scheduler = None
    if(opt.lrtype == "radam"):
        optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad],
                                    lr=opt.lr, weight_decay=opt.wd)
        scheduler = None
    if(opt.lrtype == "cyclic"):
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        scheduler = torch.optim.CyclicLR(optimizer)
    best_acc = 0.
    early_stopping_cnt = 0
    early_stopping_flag = False
    for epoch in range(opt.n_epoch):
        if not early_stopping_flag:
            # train for one epoch, valid per n batches, save the log and the best model
            if opt.rubi:
                cur_acc = train(opt, dset, (model, rubi_model), rubi_criterion, optimizer, epoch, best_acc, scheduler)
            else:
                cur_acc = train(opt, dset, model, criterion, optimizer, epoch, best_acc, scheduler)


            # remember best acc
            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= opt.max_es_cnt:
                    early_stopping_flag = True
        else:
            print(("early stop with valid acc %.4f" % best_acc))
            opt.writer.export_scalars_to_json(os.path.join(opt.results_dir, "all_scalars.json"))
            opt.writer.close()
            break  # early stop break

        if opt.debug:
            break
