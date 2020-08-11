# Tools

A variety of tools that can be used for:<br>
- Additional features of models, i.e. rubi criterion and radam optimiser.<br>
- Making various figures and visualisations from lanecheck dictionaries.<br>
- Question-type analysis.<br>

Anybody is welcome to adapt this code to whatever specific format or file layout they need. Single-use figure generation scripts are often designed with specific file structures in mind. However, we will still share our figure generation code as large portions of it will be directly useful and could be quickly adapted.<br>

## generate_h5.py
- Used for extracting regional features from raw frames into an h5 file. Format outputted from this script should be exactly compatible with out dataloader.

## inclusion_exclusion.py
- A script for use in Inclusion-Exclusion measures, IEM. 

## lanecheck_on_models.py
- Given a lanecheck dictionary specified in the accompanying bash file, consider the vote contributions of models trained on some features without some of the features, i.e. Consider SVI trained model but with the subtitle 'S' votes discounted.

## pickle3topickle2.py
- Conversion tool for turning python3 pickle objects into python2 compatible format.

## question_type.py
- A (slightly too) large file contains functionality to create a question-type dictionary from lanecheck dictionaries. This can be used to create pie charts of question-type distributions in a dataset, or to create a large heatmap demonstrating the relative performance of models on a given question type. Figure 2a.
- Use the provided bash script and argparse options to choose functionality. You will need to edit the code more fundimentally if you want to accomodate different question-type distributions than TVQA.

## radam.py
- RAdam optimiser

## rubi_criterion.py
-The RuBi criterion provided by [the official RuBi repository](https://github.com/cdancette/rubi.bootstrap.pytorch)

## save_dataset_dicts_by_qid.py
- Script to parse the provided json 'processed' dataset splits into python dictionaries.

## save_matrix_for_heatmap.py
- Given a number of lanecheck dictionaries, can create a heatmap of Intersection over Union or answer agreement between models.
- Similar to Firgure 2b.  

## validate.py
- Standalone validate implementation that can be used for pretrained models. 

## violin_plot.py
- Given a specified lanecheck dictionary in the base options, create a violin plot of answer votes using any combination of answers in the confusion matrix.
- Figure 3.

## visdom_plotter.py
- Wrapper class for a visdom plotter used in our networks. 
