#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# generate_h5.py --gpu 0 \
# --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
# --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt \
# --out test2014_resnet101_faster_rcnn_genome.tsv \
# --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
# --split tvqa

# 

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys, scandir
import base64
import numpy as np
import cv2
import h5py
from multiprocessing import Process
import random
import json
from itertools import islice
import time


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 100

def five_percent_train_test():
    # 5 percent of train
    print('5 percent of train and val')
    f = open('/home/crhf63/kable_management/data/tvqa/tvqa_train_processed.json', 'r')
    train_json = json.load(f)
    five_percent = len(train_json)/20
    #import ipdb; ipdb.set_trace()
    random.Random(2667).shuffle(train_json)
    five_percent = train_json[:five_percent]
    g = open('/home/crhf63/kable_management/data/tvqa/five_percent_tvqa_train_processed.json', 'w')
    json.dump(five_percent, g)
    f.close()
    g.close()
    print('train complete')
    # 5 percent of val
    f = open('/home/crhf63/kable_management/data/tvqa/tvqa_val_processed.json', 'r')
    val_json = json.load(f)
    five_percent = len(val_json)/20
    random.Random(2667).shuffle(val_json)
    five_percent = val_json[:five_percent]
    g = open('/home/crhf63/kable_management/data/tvqa/five_percent_tvqa_val_processed.json', 'w')
    json.dump(five_percent, g)
    f.close()
    g.close()
    print('Val complete')
    print('Ending....')



def load_image_dict(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = {}
    ######### TVQA
    if split_name == 'tvqa':
        show_paths  = [f.path for f in scandir.scandir("/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq") if f.is_dir() ] 
        for show_path in show_paths:
            clip_paths  = [g.path for g in scandir.scandir(show_path) if g.is_dir() ]
            for clip_path in clip_paths:
                temp_dict = { int(h.name.split('.')[0]):h.path for h in scandir.scandir(clip_path) if '.jpg' in h.name }
                split[clip_path] = temp_dict
    ######### 5% of TVQA Training
    elif split_name == 'tvqa_train_5p': # About 6000
        f = open('/home/crhf63/kable_management/data/tvqa/five_percent_tvqa_train_processed.json', 'r')
        five_percent = json.load(f)
        # Build a dictionary of {clip_name : { frame : ids,path }} 
        # For qa in 5 percent ......
        show_paths = {
            "Grey's Anatomy" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/grey_frames/',
            "How I Met You Mother" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/met_frames/',
            "Friends" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/friends_frames/',
            "The Big Bang Theory" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/bbt_frames/',
            "House M.D." : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/house_frames/',
            "Castle" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/castle_frames/'
        }
        for qa in five_percent:
            show_name = qa['show_name']
            clip_name = qa['vid_name']
            frame_range = qa['located_frame']
            clip_path = show_paths[show_name]+clip_name # Construct the clip path
            split = update_split(split, clip_name, frame_range[0], frame_range[1], clip_path)
    ######### 5% of TVQA Val
    elif split_name == 'tvqa_val_5p': # About 6000
        f = open('/home/crhf63/kable_management/data/tvqa/five_percent_tvqa_val_processed.json', 'r')
        five_percent = json.load(f)
        # Build a dictionary of {clip_name : { frame : ids,path }} 
        # For qa in 5 percent ......
        show_paths = {
            "Grey's Anatomy" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/grey_frames/',
            "How I Met You Mother" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/met_frames/',
            "Friends" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/friends_frames/',
            "The Big Bang Theory" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/bbt_frames/',
            "House M.D." : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/house_frames/',
            "Castle" : '/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/castle_frames/'
        }
        for qa in five_percent:
            show_name = qa['show_name']
            clip_name = qa['vid_name']
            frame_range = qa['located_frame']
            clip_path = show_paths[show_name]+clip_name # Construct the clip path
            split = update_split(split, clip_name, frame_range[0], frame_range[1], clip_path)
    ######### OTHERWISE
    else:
      print 'Unknown split'
    return split

def update_split(split, clip_name, start_frame, end_frame, clip_path): # Update the split dictionary with only the required show and frames
    frame_dict = split.get(clip_name, None)
    if start_frame == 0:    # They can't index to save their lives i have to correct for it
        start_frame = 1
    if frame_dict == None:
        frame_dict = {}
    for frame in range(start_frame, end_frame):
        frame_path = clip_path + '/' + "{:05}".format(frame) + '.jpg'     # Located frames start at 0 but frame ids start at 1? o.O, i have corrected
        frame_dict.update({frame+1 : frame_path})
    split.update({clip_path : frame_dict})    
    return split

    
def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
        'features': pool5[keep_boxes]
    }   


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--five_percent', 
                        help='Save 2667 seeded 5 percent of train and val in their own jsons',
                        default=False, type=bool)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

    
def generate_h5(gpu_id, prototxt, weights, image_dict, outfile): # image_dict(vid_name: {frame_id:frame})
    caffe.set_mode_gpu()
    caffe.set_device(int(gpu_id))
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
    if os.path.exists(outfile):
        raise FileExistsError("File exists: "+outfile)
    f = h5py.File(outfile, 'w')
    i=0 # Counter for time
    n=len(image_dict.keys())
    startt=time.time()
    for clip_path, clip_dict in image_dict.items():  # For each of the clips there are associated image paths and ids...
        clip_name = clip_path.split('/')[-1]
        grp = f.create_group(clip_name)
        for frame_id, frame_path in clip_dict.items():
            frame_group = grp.create_group(str(frame_id))
            try:
                detections = get_detections_from_im(net, frame_path, frame_id)
            except AttributeError:
                import ipdb; ipdb.set_trace()
            # { # detections
            #     'image_id': image_id,
            #     'image_h': np.size(im, 0),
            #     'image_w': np.size(im, 1),
            #     'num_boxes' : len(keep_boxes),
            #     'boxes': base64.b64encode(cls_boxes[keep_boxes]),
            #     'features': base64.b64encode(pool5[keep_boxes])
            # } f.create_dataset('1d', shape=(1,), data=42)
            frame_group.create_dataset('image_id', shape=(1,), data=detections['image_id'])
            frame_group.create_dataset('image_h', shape=(1,), data=detections['image_h'])
            frame_group.create_dataset('image_w', shape=(1,), data=detections['image_w'])
            frame_group.create_dataset('num_boxes', shape=(1,), data=detections['num_boxes'])
            frame_group.create_dataset('boxes', data=detections['boxes'])
            frame_group.create_dataset('features', data=detections['features'])
        sys.stdout.write("\r{0}%".format((float(i)/n)*100))
        sys.stdout.flush()
        i+=1
                      

if __name__ == '__main__':

    args = parse_args()
    #####################
    #import sys; sys.exit()
    #####################
    print('Called with args:')
    print(args)

    if args.five_percent:
        five_percent_train_test()
        sys.exit()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    # Multiple GPU support
    # gpu_list = gpu_id.split(',')
    # gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_dict = load_image_dict(args.data_split)
    #import ipdb; ipdb.set_trace()

    # # Split image dictionary between gpus
    # image_dicts = []
    # for x in range(len(gpus)):
    #     image_dicts.append( dict(image_dict.items()[x::len(gpus)]) )

    caffe.init_log()
    caffe.log('Using device %s' % str(gpu_id))
    generate_h5(gpu_id, args.prototxt, args.caffemodel, image_dict, args.outfile) # 74 seconds, 48MB
    # Time required:  ~ 17 days
    # Memory require: ~ 984 GB

    # procs = []    
    
    # for i,gpu_id in enumerate(gpus):
    #     outfile = '%s.%d' % (args.outfile, gpu_id)
    #     p = Process(target=generate_h5,
    #                 args=(gpu_id, args.prototxt, args.caffemodel, image_dicts[i], outfile))
    #     p.daemon = True
    #     p.start()
    #     procs.append(p)
    # for p in procs:
    #     p.join()            
                  
