__author__ = "Jie Lei"
__edited__ = "Jumperkables"
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import load_pickle, save_pickle, load_json, files_exist

# For bert
from transformers import BertTokenizer

class TVQADataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.raw_train = load_json(opt.train_path)
        self.raw_test = load_json(opt.test_path)
        self.raw_valid = load_json(opt.valid_path)
        self.vfeat_load = opt.vid_feat_flag
        self.reg_flag = "regional" in opt.input_streams
        self.regtopk_flag = (-1 != opt.regional_topk)

        # Options are useful to access
        self.opt = opt

        # Regional features loading
        if self.reg_flag:
            self.reg_h5 = h5py.File(opt.reg_feat_path, "r", driver=None)
        # Visual concept loaded
        if not opt.my_vcpt: # Their visual concepts
            self.vcpt_dict = load_pickle(opt.vcpt_path)
        else:
            # Load the visual concepts classes i got
            with open('/home/jumperkables/regional_stuff/faster-rcnn.pytorch/data/pretrained_model/objects_vocab.txt', 'r') as f:
                data = f.readlines()
            pascal_classes = np.asarray(['__background__'])
            pascal_classes = np.append(pascal_classes, np.asarray(data))
            self.vcpt_classes = [x.strip('\n') for x in pascal_classes] 
            if not self.reg_flag: # even if we're not using regional features, we still need that file for my classes
                self.reg_h5 = h5py.File(opt.reg_feat_path, "r", driver=None)

        # Video features
        if self.vfeat_load:
            self.vid_h5 = h5py.File(opt.vid_feat_path, "r", driver=None)#opt.h5driver)

        # Get bert model ready
        if opt.bert == "default":
            self.bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        elif opt.bert == "multi_choice":
            self.bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        elif opt.bert == "qa":
            self.bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')


        self.glove_embedding_path = opt.glove_path
        self.normalize_v = opt.normalize_v
        self.with_ts = opt.with_ts
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

        # set word embedding / vocabulary
        self.embedding_dim = opt.embedding_size
        if  opt.bert is None:
            self.word2idx_path = opt.word2idx_path
            self.idx2word_path = opt.idx2word_path
            self.vocab_embedding_path = opt.vocab_embedding_path
            self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
            self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
            self.offset = len(self.word2idx)

        # set entry keys
        if self.with_ts:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "located_sub_text"]
        else:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub_text"]
        self.vcpt_key = "vcpt"
        self.label_key = "answer_idx"
        self.qid_key = "qid"
        self.vid_name_key = "vid_name"
        self.located_frm_key = "located_frame"
        for k in self.text_keys + [self.vcpt_key, self.qid_key, self.vid_name_key]:
            if k == "vcpt":
                continue
            assert k in list(self.raw_valid[0].keys())

        # build/load vocabulary
        if  opt.bert is None:
            if not files_exist([self.word2idx_path, self.idx2word_path, self.vocab_embedding_path]):
                print("\nNo cache founded.")
                self.build_word_vocabulary(word_count_threshold=opt.word_count_threshold)
            else:
                print("\nLoading cache ...")
                self.word2idx = load_pickle(self.word2idx_path)
                self.idx2word = load_pickle(self.idx2word_path)
                self.vocab_embedding = load_pickle(self.vocab_embedding_path)

    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            return self.raw_test

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        items = []
        if self.with_ts:
            cur_start, cur_end = self.cur_data_dict[index][self.located_frm_key]
        cur_vid_name = self.cur_data_dict[index][self.vid_name_key]
        #if cur_vid_name == 'friends_s10e17-18_seg02_clip_16':
        #    import ipdb; ipdb.set_trace()

        # add text keys
        if self.opt.bert is None:
            for k in self.text_keys:
                items.append(self.numericalize(self.cur_data_dict[index][k]))
        else:
            for k in self.text_keys:
                items.append(self.bert_tokeniser.encode(self.cur_data_dict[index][k], add_special_tokens=True))


        # add vcpt
        if(self.opt.my_vcpt):   # use my visual concepts? # my_vcpt
            # TEST IF MY VISUAL CONCEPTS ARE LOADED INTO THEIR CLASSES PROPERLY
            # self.vcpt_classes
            if self.with_ts:
                if cur_start == 0:
                    cur_start = 1
                max_idx = max([int(frm) for frm in self.reg_h5[cur_vid_name]['classes'].keys()])
                cur_vis_sen = [self.reg_h5[cur_vid_name]['classes'][str(frame_indx)] for frame_indx in range(cur_start, min(cur_end, max_idx)) ]
                cur_vis_sen = [ [ obj for obj in frame ] for frame in cur_vis_sen ]  # Get all of the class indexes
                for j, frame in enumerate(cur_vis_sen):
                    for xx in range(len(frame)):
                        frame[xx] = self.vcpt_classes[int(frame[xx])]
                    cur_vis_sen[j] = " , ".join(frame)
            else:  # handled by vid_path
                raise Exception("Not implemented regional without ts")
            # if self.normalize_v:
            #     cur_reg_feat = nn.functional.normalize(cur_reg_feat, p=2, dim=1)
        else:                   # use their visual concepts
            if self.with_ts:
                cur_vis_sen = self.vcpt_dict[cur_vid_name][cur_start:cur_end + 1]
            else:
                cur_vis_sen = self.vcpt_dict[cur_vid_name]
        if self.opt.bert is not None:   # If using bert
            #import ipdb; ipdb.set_trace()
            cur_vis_sen = " , ".join(cur_vis_sen)
            cur_vis_sen = self.vcpt_bert(cur_vis_sen)
            cur_vis_sen = self.bert_tokeniser.encode(cur_vis_sen, add_special_tokens=True)
            items.append(cur_vis_sen)
        else:
            #import ipdb; ipdb.set_trace()
            cur_vis_sen = " , ".join(cur_vis_sen)
            items.append(self.numericalize_vcpt(cur_vis_sen))


        # # add other keys
        if self.mode == 'test':
            items.append(666)  # this value will not be used
        else:
            items.append(int(self.cur_data_dict[index][self.label_key]))
        for k in [self.qid_key]:
            items.append(self.cur_data_dict[index][k])
        items.append(cur_vid_name)


        # add visual feature
        if self.vfeat_load:
            if self.with_ts:
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][cur_start:cur_end])
            else:  # handled by vid_path
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][:480])
            if self.normalize_v:
                cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)
        else:
            cur_vid_feat = torch.zeros([2, 2])  # dummy placeholder
        items.append(cur_vid_feat)


        # add regional feature
        if self.reg_flag and not self.regtopk_flag: # here regional feature
            if self.with_ts:
                if cur_start == 0:
                    cur_start = 1
                max_idx = max([int(frm) for frm in self.reg_h5[cur_vid_name]['features'].keys()])
                cur_reg_feat = [torch.from_numpy(np.array(self.reg_h5[cur_vid_name]['features'][str(frame_indx)])) for frame_indx in range(cur_start, min(cur_end, max_idx)) ]
                cur_reg_feat = torch.stack(cur_reg_feat)
            else:  # handled by vid_path
                raise Exception("regional Not implemented regional without ts")
            if self.normalize_v:
                cur_reg_feat = nn.functional.normalize(cur_reg_feat, p=2, dim=1)
            ######## Tester
            #import ipdb; ipdb.set_trace()
            ########
        else:
            cur_reg_feat = torch.zeros([2, 2, 2])  # dummy placeholder
        items.append(cur_reg_feat)

        # add topk regional feature
        if self.regtopk_flag and self.reg_flag: # here regional feature
            if self.with_ts:
                if cur_start == 0:
                    cur_start = 1
                max_idx = max([int(frm) for frm in self.reg_h5[cur_vid_name]['features'].keys()])
                cur_regtopk_feat = []
                cur_regtopk_scores = []
                for frame_indx in range(cur_start, min(cur_end, max_idx)):                  
                    to_add_feat = torch.from_numpy(np.array(self.reg_h5[cur_vid_name]['features'][str(frame_indx)]))
                    to_add_score = torch.from_numpy(np.array(self.reg_h5[cur_vid_name]['scores'][str(frame_indx)]))
                    if to_add_feat.shape[0] != 20:
                        to_do = 20-to_add_feat.shape[0]
                        for yy in range(to_do):                 # If there aren't 20 detections per frame pad with zero detections
                            to_add_feat = torch.cat((to_add_feat, torch.zeros(1, 2048)), dim=0)
                            to_add_score = torch.cat((to_add_score, torch.zeros(1)), dim=0)
                    cur_regtopk_feat.append(to_add_feat)
                    cur_regtopk_scores.append(to_add_score)
                if cur_regtopk_feat == []:  # If no detections in the specified range return nothing
                    cur_regtopk_feat = torch.zeros(2, 2048)
                else:
                    cur_regtopk_feat = torch.stack(cur_regtopk_feat)
                    cur_regtopk_scores = torch.stack(cur_regtopk_scores)

                    cur_regtopk_feat = torch.flatten(cur_regtopk_feat, end_dim=1)    # order the tensors by score
                    cur_regtopk_scores = torch.flatten(cur_regtopk_scores, end_dim=1)    # order the tensors by score
                    
                    cur_regtopk_scores, order = torch.sort(cur_regtopk_scores, 0, True)
                    cur_regtopk_feat = cur_regtopk_feat[order]
                    cur_regtopk_feat = cur_regtopk_feat[:self.opt.regional_topk]              
            else:  # handled by vid_path
                raise Exception("Top-k regional Not implemented regional without ts")
            if self.normalize_v:
                cur_regtopk_feat = nn.functional.normalize(cur_regtopk_feat, p=2, dim=1)
        else:
            cur_regtopk_feat = torch.zeros([2, 2])  # dummy placeholder
        items.append(cur_regtopk_feat)

        return items

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        # !!!! remove comma here, since they are too many of them
        words = [w for w in words if w != ","]
        words = words + [eos_word] if eos else words
        return words

    def numericalize(self, sentence, eos=True):
        """convert words to indices"""
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
        return sentence_indices

    def numericalize_vcpt(self, vcpt_sentence):
        """convert words to indices, additionally removes duplicated attr-object pairs"""
        attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
        unique_pairs = []
        for pair in attr_obj_pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        words.append("<eos>")
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in words]
        return sentence_indices

    def vcpt_bert(self, vcpt_sentence):
        """convert words to indices, additionally removes duplicated attr-object pairs"""
        attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
        unique_pairs = []
        for pair in attr_obj_pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        words.append("")
        return words

    @classmethod
    def load_glove(cls, filename):
        """ Load glove embeddings into a python dict
        returns { word (str) : vector_embedding (torch.FloatTensor) }"""
        glove = {}
        with open(filename) as f:
            for line in f.readlines():
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def build_word_vocabulary(self, word_count_threshold=0):
        """borrowed this implementation from @karpathy's neuraltalk."""
        print("Building word vocabulary starts.\n")
        all_sentences = []
        for k in self.text_keys:
            all_sentences.extend([ele[k] for ele in self.raw_train])

        word_counts = {}
        for sentence in all_sentences:
            for w in self.line_to_words(sentence, eos=False, downcase=True):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in list(self.word2idx.keys())]
        print(("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" %
              (len(vocab), word_count_threshold)))

        # build index and vocabularies
        for idx, w in enumerate(vocab):
            self.word2idx[w] = idx + self.offset
            self.idx2word[idx + self.offset] = w
        print(("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word))))

        # Make glove embedding.
        print(("Loading glove embedding at path : %s. \n" % self.glove_embedding_path))
        glove_full = self.load_glove(self.glove_embedding_path)
        print("Glove Loaded, building word2idx, idx2word mapping. This may take a while.\n")
        glove_matrix = np.zeros([len(self.idx2word), self.embedding_dim])
        glove_keys = list(glove_full.keys())
        for i in tqdm(list(range(len(self.idx2word)))):
            w = self.idx2word[i]
            w_embed = glove_full[w] if w in glove_keys else np.random.randn(self.embedding_dim) * 0.4
            glove_matrix[i, :] = w_embed
        self.vocab_embedding = glove_matrix
        print(("Vocab embedding size is :", glove_matrix.shape))

        print("Saving cache files ...\n")
        save_pickle(self.word2idx, self.word2idx_path)
        save_pickle(self.idx2word, self.idx2word_path)
        save_pickle(glove_matrix, self.vocab_embedding_path)
        print("Building  vocabulary done.\n")


class Batch(object):
    def __init__(self):
        self.__doc__ = "empty initialization"

    @classmethod
    def get_batch(cls, keys=None, values=None):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        assert keys is not None and values is not None
        for k, v in zip(keys, values):
            setattr(batch, k, v)
        return batch


def pad_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    def pad_sequences(sequences):
        sequences = [torch.LongTensor(s) for s in sequences]
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq[:end]
        return padded_seqs, lengths

    def pad_video_sequences(sequences):
        """sequences is a list of torch float tensors (created from numpy)"""
        #import ipdb; ipdb.set_trace()
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        v_dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), v_dim).float()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
        return padded_seqs, lengths
    
    def pad_regional_sequences(sequences):
        """sequences is a list of torch float tensors created by jumperkables"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded_seqs = torch.zeros(len(sequences), max(lengths), sequences[0].shape[1], sequences[0].shape[2]).float()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
        return padded_seqs, lengths

    def pad_regtopk_sequences(sequences):
        """Implemented by jumperkables"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded_seqs = torch.zeros(len(sequences), max(lengths), sequences[0].shape[1]).float()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
        return padded_seqs, lengths

    # separate source and target sequences
    column_data = list(zip(*data))
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_name_key = "vid_name"
    vid_feat_key = "vid"
    reg_feat_key = "reg"
    reg_topk_key = "regtopk"
    all_keys = text_keys + [label_key, qid_key, vid_name_key, vid_feat_key, reg_feat_key, reg_topk_key]
    all_values = []
    for i, k in enumerate(all_keys):
        if k in text_keys:
            all_values.append(pad_sequences(column_data[i]))
        elif k == label_key:
            all_values.append(torch.LongTensor(column_data[i]))
        elif k == vid_feat_key:
            all_values.append(pad_video_sequences(column_data[i]))
        elif k == reg_feat_key:
            all_values.append(pad_regional_sequences(column_data[i]))
        elif k == reg_topk_key:
            all_values.append(pad_regtopk_sequences(column_data[i]))
        else:
            all_values.append(column_data[i])

    batched_data = Batch.get_batch(keys=all_keys, values=all_values)
    return batched_data


def preprocess_inputs(batched_data, max_sub_l, max_vcpt_l, max_vid_l, device="cuda:0"):
    """clip and move to target device"""
    # imagenet in preprocessing to figure out how to do the same for regional feautes
    max_len_dict = {"sub": max_sub_l, "vcpt": max_vcpt_l, "vid": max_vid_l, "reg":max_vid_l}
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_feat_key = "vid"
    reg_feat_key = "reg"
    regtopk_feat_key = "regtopk"
    model_in_list = []
    for k in text_keys + [vid_feat_key, reg_feat_key, regtopk_feat_key]:
        #import ipdb; ipdb.set_trace()
        v = getattr(batched_data, k)
        if k in max_len_dict:
            ctx, ctx_l = v
            max_l = min(ctx.size(1), max_len_dict[k])
            if ctx.size(1) > max_l:
                ctx_l = ctx_l.clamp(min=1, max=max_l)
                ctx = ctx[:, :max_l]
            model_in_list.extend([ctx.to(device), ctx_l.to(device)])
        else:
            model_in_list.extend([v[0].to(device), v[1].to(device)])
    target_data = getattr(batched_data, label_key)
    target_data = target_data.to(device)
    qid_data = getattr(batched_data, qid_key)
    return model_in_list, target_data, qid_data


if __name__ == "__main__":
    # python tvqa_dataset.py --input_streams sub
    import sys
    from config import BaseOptions
    sys.argv[1:] = ["--input_streams", "vcpt", "imagenet"]
    opt = BaseOptions().parse()

    dset = TVQADataset(opt, mode="train")
    data_loader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=pad_collate)
    for idx, batch in enumerate(data_loader):
        pass



