import os
import sys
sys.path.insert(1, "..")
#sys.path.insert(1, os.path.expanduser("~/kable_management/projects/tvqa_modality_bias"))
#sys.path.insert(1, os.path.expanduser("~/kable_management/mk8+-tvqa"))
from utils import load_pickle, save_pickle, load_json, files_exist

# Turn the json processed dataset into a dictionary and it
def dset2dict(dset, name):
    dset_dict={}
    for idx, question in enumerate(dset):
        dset_dict[question['qid']] = question
    save_pickle(dset_dict, os.path.expanduser("~/kable_management/data/tvqa/"+name+"_dict.pickle"))
    print(name, "Done")



if __name__ == "__main__":
    train_dset = load_json(os.path.expanduser("~/kable_management/data/tvqa/tvqa_train_processed.json"))
    val_dset = load_json(os.path.expanduser("~/kable_management/data/tvqa/tvqa_val_processed.json"))
    test_dset = load_json(os.path.expanduser("~/kable_management/data/tvqa/tvqa_test_public_processed.json"))
    dset2dict(train_dset, "train")
    dset2dict(val_dset, "val")
    dset2dict(test_dset, "test")
    total_dset = []
    total_dset += train_dset
    total_dset += val_dset
    total_dset += test_dset
    dset2dict(total_dset ,'total')
