__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import torch.nn as nn
from DNC.dnc  import DNC
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class bAbI_dnc(nn.Module):
    def __init__(self, cfg):
        super(bAbI_dnc, self).__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(self.cfg["hyperparameters"]["vocab_size"],
                                      self.cfg["hyperparameters"]["embedding_dim"])
        self.tanh = nn.Tanh()
        self.rnn = DNC(
            input_size=cfg["dnc"]["input_size"],
            output_size=cfg["dnc"]["output_size"],
            hidden_size=cfg["dnc"]["hidden_dim"],
            rnn_type=cfg["dnc"]["rnn_type"],
            num_layers=cfg["dnc"]["num_layers"],
            num_hidden_layers=cfg["dnc"]["num_layers_hidden"],
            dropout=cfg["hyperparameters"]["dropout"],
            nr_cells=cfg["dnc"]["n"],
            cell_size=cfg["dnc"]["w"],
            read_heads=cfg["dnc"]["r"],
            gpu_id=cfg["hyperparameters"]["gpu_id"],
            debug=cfg["dnc"]["debug"],
            batch_first=True,
            independent_linears=True)

    def forward(self, inputs, chx, mhx, rv, pass_through_memory, reset_experience):

        if self.cfg["hyperparameters"]["use_one_hot"]:
            if self.cfg["dnc"]["debug"]:
                logits, (chx, mhx, rv), v = self.rnn(inputs,
                                                (None, mhx, None),
                                                reset_experience=reset_experience,
                                                pass_through_memory=pass_through_memory)
            else:
                logits, (chx, mhx, rv)    = self.rnn(inputs,
                                                (None, mhx, None),
                                                reset_experience=reset_experience,
                                                pass_through_memory=pass_through_memory)
        else:
            inputs = self.embedding(inputs)
            inputs = self.tanh(inputs)
            # packed_input = pack_padded_sequence(inputs, batch_first=True)
            if self.cfg["dnc"]["debug"]:
                logits, (chx, mhx, rv), v = self.rnn(inputs,
                                                (None, mhx, None),
                                                reset_experience=reset_experience,
                                                pass_through_memory=pass_through_memory)
            else:
                logits, (chx, mhx, rv)    = self.rnn(inputs,
                                                (None, mhx, None),
                                                reset_experience=reset_experience,
                                                pass_through_memory=pass_through_memory)
        if self.cfg["dnc"]["debug"]:
            return logits, (chx, mhx, rv), v
        else:
            return logits, (chx, mhx, rv)


class bAbI_lstm(nn.Module):
    def __init__(self, cfg):
        super(bAbI_lstm, self).__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(self.cfg["hyperparameters"]["vocab_size"],
                                      self.cfg["hyperparameters"]["embedding_dim"])
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(cfg["dnc"]["input_size"],
                            cfg["lstm"]["hidden_dim"],
                            cfg["lstm"]["num_layers"],
                            batch_first=True)
        self.fc   = nn.Linear(cfg["lstm"]["hidden_dim"], cfg["dnc"]["output_size"])

    def forward(self, inputs):
        if self.cfg["hyperparameters"]["use_one_hot"]:
            o, (_, _) = self.lstm(inputs)
        else:
            inputs = self.embedding(inputs)
            inputs = self.tanh(inputs)
            o, (_, _) = self.lstm(inputs)
        logits = self.fc(o)
        return logits
