import pickle

in_path = "/home/jumperkables/kable_management/data/tvqa/"+"cache/"+"word2idx.pickle"
out_path = "/home/jumperkables/kable_management/data/tvqa/"+"cache/"+"py2_word2idx.pickle"
with open(in_path, "rb") as f:
    w = pickle.load(f)

pickle.dump(w, open(out_path,"wb"), protocol=2)