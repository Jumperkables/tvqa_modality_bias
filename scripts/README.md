# Scripts

## If you're looking at the A-vs-C branch, you can disregard these scripts

Some example scripts for running various experiments. We capture modality votes using a 'lanecheck' system. Visdom integration is supported.

A few steps to observe:<br>

0. Start up a [visdom server](https://pypi.org/project/visdom/) (included in requirements).<br>

1. In `tools/visdom_plotter.py`, edit the `__init__` function and link to your own visdom server. (Feel free to remove all visdom functionality from `main.py` if you don't care).<br>


# Baseline

0. Run the baseline models seen in Table 1a. Choose the feature streams to activate with the `--input_streams`.<br> 
1. Running models will save a 'lanecheck dictionary', which counts the votes for each question from each feature stream enabled. Make sure the `--lanecheck_path` option is set somewhere useful, most sensibly inside the `--results_dir_base`. Halt this functionality by changing `--lanecheck` to False.<br>
2. Regional features are implemented a little differently. To use regional features, add 'regional' to `--input_streams` <strong>AND</strong> add `--regional_topk X` where X is the number of regional features across the segment to consider. Example shown in sr_bert.sh.<br>
3. To run BERT models, add the flag `--bert default`, and to run GloVe models, simply omit that same BERT flag.<br>


# Dual Stream
 
0. Lanechecking is not implemented for dual stream models, and dual stream considers subtitle and imagenet streams only. `--dual_stream` option controls this behaviour.
1. The included scripts are exactly the experiments ran.

# RuBi

0. The included script is the RuBi implementation of SI with weighting of 1 as in our paper. `--rubi` option controls this behaviour.
