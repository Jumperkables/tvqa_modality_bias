# Modality Bias in TVQA  
The official github repository for the paper ["On Modality Bias in the TVQA Dataset"](pls.accept.co.uk)
## TVQA:
Our framework is built and adapted from the [official TVQA repository](https://github.com/jayleicn/TVQA). This repository includes access to the original dataset, the official website, the submission leaderboard and other projects, including TVQA+.

## Modality Data Subsets:

Using the IEM inclusion-exclusion measure in our paper, we propose subsets that respond to a mixture of modalities and features. 

## Using our framework:

The essence of our framework can be used for <strong>any</strong> video-QA dataset with appropriate features. You'll have to adapt at least the dataloader and model classes to fit your new dataset. They function almost identically to the baseline TVQA classes, with added functionality. You may find it helpful to replicate our TVQA experiments first:

0. `git clone `https://github.com/Jumperkables/tvqa_modality_bias`
1. `pip install -r requirements.txt`
2. Now assemble the dataset to run:
3. Install the [pytorch block fusion package](https://github.com/Cadene/block.bootstrap.pytorch), and place it in this directory. You will need to edit imports in the `model/tvqa_abc_bert_nofc.py` file to accomodate this fusion package for bilinear pooling.

## Data:

### Questions, Answers, Subtitles and ImageNet: 

Clone the [TVQA github repository](https://github.com/jayleicn/TVQA) and follow steps 1, 2 and 3 for data extraction. This will give you the processed json files for the validation and training set. The processed json files contain questions, answers and subtitles. ImageNet features are in an h5 file. The ImageNet file is large and will require a significant amount of memory to load into memory, but you can specify no core driver for loading for lazy reads to avoid this.

### Visual Concepts:

Visual concepts are contained in `det_visual_concepts_hq.pickle` file.


### Regional Features: 

There are at most 20 regional features per frame, each 2048d, making this far too big to share. The original TVQA repository doesn't supply regional features or support them in the dataloader. We have implemented regional features seen in our paper under the name `regional_topk` (not `regional`). <br>
You will need to follow the instruction [here](http://tvqa.cs.unc.edu/download_tvqa.html), and apply for the raw TVQA video frames, and extract them yourself.<br>
Specifically, follow instructions from [here](https://github.com/peteanderson80/bottom-up-attention#demo). Once you have set up this repository, add our `tools/generate_h5.py` from our repository to the `bottom-up-attention/tools/` directory. Adapt this file to your raw video file location and run, extracting an h5 file for the entire dataset of frames (In our scripts we have called our regional file 100.h5). It will take a while, but our generation script should help a lot, and shows you the exact structure our dataloader will expect form the h5 file.<br>

<strong>See our `example_data_directory` as a guideline.</strong>


## Scripts:

Scripts to run our experiments after data is collected, edit the relevant dataset and import paths in the main, config, utils and tvqa_dataset files to suit your repository structure and run these scripts.

## Tools:

Some tools used in our experiments for visualisation and convenience.

## Citation:
Published at BMVC 2020<br><br>
@inproceedings{mbintvqa,<br>
  title={On Modality Bias in the TVQA Dataset},<br>
  author={Winterbottom, T. and Xiao, S. and McLean, A. and Al Moubayed, N.},<br>
  booktitle={Proceedings of the British Machine Vision Conference ({BMVC})},<br>
  year={2020}<br>
}

## Help:
Feel free to contact me @ `thomas.i.winterbottom@durham.ac.uk` if you have any criticisms you'd like me to hear out or would like any help
