__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import torch
import pickle
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.utils.data.dataset import Dataset


class bAbI_Dataset(Dataset):
    def __init__(self, data_path, lexicon_path, use_one_hot):
        super(bAbI_Dataset, self).__init__()
        self.data = self.load(data_path)
        self.lexicon = self.load(lexicon_path)
        self.idx_2_words = {
            v:k for (k,v) in self.lexicon.items()
        }
        self.len_lexicon = len(self.lexicon)
        self.use_one_hot = use_one_hot
        # Ignore all classes expect "-"
        self.loss_weight = torch.zeros((self.len_lexicon, ), dtype=torch.float)
        self.loss_weight[self.lexicon["-"]] = 1.0
        self.pad_idx = self.lexicon["*"]
        self.task = data_path.split("/")[-1].split("_")[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        input_vec = np.array(sample["inputs"], dtype=np.float32)
        input_raw = " ".join([w for w in list(map(lambda i: self.idx_2_words[i], sample["inputs"]))])
        output_raw = " ".join([w for w in list(map(lambda i: self.idx_2_words[i], sample["outputs"]))])
        output_vec = np.array(sample["inputs"], dtype=np.float32)

        mask_answer = input_vec == self.lexicon["-"]
        # pad_idx = np.argwhere(input_vec == self.lexicon["*"])[0][0]
        answer_idx = []
        for i in range(len(sample["inputs"])):
            if sample["inputs"][i] == self.lexicon["-"]:
                answer_idx.append(i)
        output_vec[mask_answer] = sample["outputs"]
        answer_idx += (13 - len(answer_idx)) * [-1]
        answer_gt = deepcopy(sample["outputs"])
        answer_gt += (13 - len(answer_gt)) * [-1]

        if self.use_one_hot:
            input_vec  = np.array([self.one_hot(int(idx), self.len_lexicon) for idx in input_vec])
        else:
            input_vec  = np.array(input_vec, dtype=np.int64)
        output_vec_1_hot = np.array([self.one_hot(int(idx), self.len_lexicon) for idx in output_vec])
        output_vec = np.array(output_vec, dtype=np.int64)
        weights_vec = np.ones_like(input_vec, dtype=np.float32)
        mask_answer = 1.0 * mask_answer
        # weights_vec[pad_idx:, :] = 0.0

        batch = {
            "input"           : torch.from_numpy(input_vec),
            "output"          : torch.from_numpy(output_vec),
            "input_raw"       : input_raw,
            "output_raw"      : output_raw,
            "output_vec_1_hot": torch.from_numpy(output_vec_1_hot),
            "weights"         : torch.from_numpy(weights_vec),
            "answers_idx"     : torch.tensor(answer_idx).long(),
            "answers_gt"      : torch.tensor(answer_gt).long(),
            "mask_answer"     : torch.tensor(mask_answer),
            "length_answers"  : torch.tensor(sample["length_answers"]),
            # "answers_gt"    : sample["answers_gt"],
            "tasks"           : sample["task"]}
        return batch

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def one_hot(idx, size):
        res = np.zeros(size, dtype=np.float32)
        res[idx] = 1.0
        return res
