from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.models as models
from DNC.dnc_mlp import DNC_MLP
from DNC.dnc import DNC


class ImgEncoder(nn.Module):
    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features  # input size of feature vector
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    # remove last fc layer

        self.model = model                              # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)    # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]
        return qst_feature


class QstEncoderDnc(nn.Module):
    def __init__(self, cfg):

        super(QstEncoderDnc, self).__init__()
        self.cfg = cfg
        self.word2vec = nn.Embedding(cfg["hyperparameters"]["qst_vocab_size"], cfg["hyperparameters"]["embedding_dim"])
        self.tanh = nn.Tanh()
        # self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.dnc_q = DNC(
            input_size=cfg["hyperparameters"]["embedding_dim"],
            output_size=cfg["hyperparameters"]["commun_embed_size"],
            hidden_size=cfg["dnc_q"]["hidden_dim"],
            rnn_type=cfg["dnc_q"]["rnn_type"],
            num_layers=cfg["dnc_q"]["num_layers"],
            num_hidden_layers=cfg["dnc_q"]["num_layers_hidden"],
            bias=True,
            batch_first=True,
            dropout=cfg["dnc_q"]["dropout"],
            bidirectional=False,
            nr_cells=cfg["dnc_q"]["n"],
            cell_size=cfg["dnc_q"]["w"],
            read_heads=cfg["dnc_q"]["r"],
            gpu_id=cfg["hyperparameters"]["gpu_id"],
            independent_linears=True,
            share_memory=True,
            debug=cfg["dnc_q"]["debug"],
            clip=20,
        )
        # self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question, chx=None, mhx=None, rv=None, pass_through_memory=True, reset_experience=False):
        qst_vec = self.word2vec(question)  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        v = None
        if self.cfg["dnc_q"]["debug"]:
            qst_feature, (chx, mhx, rv), v = self.dnc_q(
                qst_vec,
                (None, mhx, None),
                reset_experience=reset_experience,
                pass_through_memory=pass_through_memory)
        else:
            qst_feature, (chx, mhx, rv) = self.dnc_q(
                qst_vec,
                (None, mhx, None),
                reset_experience=reset_experience,
                pass_through_memory=pass_through_memory)
        qst_feature = qst_feature.mean(1)
        return qst_feature, (chx, mhx, rv), v


class VqaModelDncC(nn.Module):
    def __init__(self, cfg):  # embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModelDncC, self).__init__()
        self.cfg = cfg
        self.img_encoder = ImgEncoder(cfg["hyperparameters"]["commun_embed_size"])
        self.qst_encoder = QstEncoder(
            cfg["hyperparameters"]["qst_vocab_size"],
            cfg["hyperparameters"]["embedding_dim"],
            cfg["hyperparameters"]["commun_embed_size"],
            cfg["lstm"]["num_layers"],
            cfg["lstm"]["hidden_dim"])
        if cfg["dnc_c"]["nonlinearity"] == "tanh":
            self.nonlinearity = nn.Tanh()
        elif cfg["dnc_c"]["nonlinearity"] == "relu":
            self.nonlinearity = nn.ReLU()
        elif cfg["dnc_c"]["nonlinearity"] == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        else:
            raise ValueError("<{}> is not a valid non-linearity function.".format(cfg["dnc_c"]["nonlinearity"]))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        if cfg["dnc_c"]["type"] == "MLP":
            self.dnc = DNC_MLP(
                input_size=cfg["hyperparameters"]["commun_embed_size"],
                output_size=cfg["dnc_c"]["output_size"],
                hidden_size=cfg["dnc_c"]["hidden_dim"],
                num_hidden_layers=cfg["dnc_c"]["num_layers_hidden"],
                dropout=cfg["dnc_c"]["dropout"],
                nr_cells=cfg["dnc_c"]["n"],
                cell_size=cfg["dnc_c"]["w"],
                read_heads=cfg["dnc_c"]["r"],
                nonlinearity=self.nonlinearity,
                gpu_id=cfg["hyperparameters"]["gpu_id"],
                debug=cfg["dnc_c"]["debug"],
                clip=20,
            )
        elif cfg["dnc_c"]["type"] == "LSTM":
            self.dnc = DNC(
                input_size=cfg["hyperparameters"]["commun_embed_size"],
                output_size=cfg["dnc_c"]["output_size"],
                hidden_size=cfg["dnc_c"]["hidden_dim"],
                rnn_type=cfg["dnc_c"]["rnn_type"],
                num_layers=cfg["dnc_c"]["num_layers"],
                num_hidden_layers=cfg["dnc_c"]["num_layers_hidden"],
                bias=True,
                batch_first=True,
                dropout=cfg["dnc_c"]["dropout"],
                bidirectional=cfg["dnc_c"]["bidirectional"],
                nr_cells=cfg["dnc_c"]["n"],
                cell_size=cfg["dnc_c"]["w"],
                read_heads=cfg["dnc_c"]["r"],
                gpu_id=cfg["hyperparameters"]["gpu_id"],
                independent_linears=True,
                share_memory=True,
                debug=cfg["dnc_c"]["debug"],
                clip=20)
        else:
            raise ValueError("dnc controller type <{}> is not defined".format(cfg["dnc"]["dnc_c_type"]))

        if cfg["dnc_c"]["concat_out_rv"]:
            in_fc_1 = cfg["dnc_c"]["output_size"] + cfg["dnc_c"]["w"] * cfg["dnc_c"]["r"]
        else:
            in_fc_1 = cfg["dnc_c"]["output_size"]
        self.fc_1 = nn.Linear(in_fc_1, cfg["hyperparameters"]["ans_vocab_size"])
        self.fc_2 = nn.Linear(cfg["hyperparameters"]["ans_vocab_size"], cfg["hyperparameters"]["ans_vocab_size"])

    def forward(self, img, qst, chx_c=None, mhx_c=None, rv_c=None):
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        # combined_feature = combined_feature.unsqueeze(1)
        v_c = None
        if self.cfg["dnc_c"]["type"] == "MLP":
            if self.cfg["dnc_c"]["debug"]:
                combined_feature, (mhx_c, rv_c), v_c = self.dnc(
                    combined_feature, mhx_c, rv_c)
            else:
                combined_feature, (mhx_c, rv_c) = self.dnc(
                    combined_feature, mhx_c, rv_c)
            if self.cfg["dnc_c"]["concat_out_rv"]:
                last_feat = torch.cat((combined_feature, rv_c), 1)
            else:
                last_feat = combined_feature
            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            last_feat = self.fc_1(last_feat)
            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            logits    = self.fc_2(last_feat)
            return logits, (mhx_c, rv_c), v_c

        elif self.cfg["dnc_c"]["type"] == "LSTM":
            combined_feature = combined_feature.unsqueeze(1)
            if self.cfg["dnc_c"]["debug"]:
                combined_feature, (chx_c, mhx_c, rv_c), v_c = self.dnc(
                    combined_feature, (None, mhx_c, None))
            else:
                combined_feature, (chx_c, mhx_c, rv_c) = self.dnc(
                    combined_feature, (None, mhx_c, None))

            combined_feature = combined_feature.squeeze(1)
            if self.cfg["dnc_c"]["concat_out_rv"]:
                last_feat = torch.cat((combined_feature, rv_c), 1)
            else:
                last_feat = combined_feature

            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            last_feat = self.fc_1(last_feat)
            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            logits    = self.fc_2(last_feat)
            return logits, (chx_c, mhx_c, rv_c), v_c


class VqaModelDncQ(nn.Module):
    def __init__(self, cfg):  # embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModelDncQ, self).__init__()
        self.cfg = cfg
        self.img_encoder = ImgEncoder(cfg["hyperparameters"]["commun_embed_size"])
        self.qst_encoder = QstEncoderDnc(cfg)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(cfg["hyperparameters"]["commun_embed_size"], cfg["hyperparameters"]["ans_vocab_size"])
        self.fc2 = nn.Linear(cfg["hyperparameters"]["ans_vocab_size"], cfg["hyperparameters"]["ans_vocab_size"])

    def forward(self, img, qst, chx=None, mhx=None, rv=None):
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature, (chx, mhx, rv), v = self.qst_encoder(
            qst, chx=chx, mhx=mhx, rv=rv)

        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]
        return combined_feature, (chx, mhx, rv), v


class VqaModel(nn.Module):
    def __init__(self, cfg):  # embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        self.cfg = cfg
        self.img_encoder = ImgEncoder(cfg["hyperparameters"]["commun_embed_size"])
        self.qst_encoder = QstEncoder(
            cfg["hyperparameters"]["qst_vocab_size"],
            cfg["hyperparameters"]["embedding_dim"],
            cfg["hyperparameters"]["commun_embed_size"],
            cfg["lstm"]["num_layers"],
            cfg["lstm"]["hidden_dim"])

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(cfg["hyperparameters"]["commun_embed_size"], cfg["hyperparameters"]["ans_vocab_size"])
        self.fc2 = nn.Linear(cfg["hyperparameters"]["ans_vocab_size"], cfg["hyperparameters"]["ans_vocab_size"])

    def forward(self, img, qst):
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]
        return combined_feature


class VqaModelDncQC(nn.Module):
    def __init__(self, cfg):
        super(VqaModelDncQC, self).__init__()
        self.cfg = cfg
        self.img_encoder = ImgEncoder(cfg["hyperparameters"]["commun_embed_size"])
        self.qst_encoder = QstEncoderDnc(cfg)
        if cfg["dnc_c"]["nonlinearity"] == "tanh":
            self.nonlinearity = nn.Tanh()
        elif cfg["dnc_c"]["nonlinearity"] == "relu":
            self.nonlinearity = nn.ReLU()
        elif cfg["dnc_c"]["nonlinearity"] == "sigmoid":
            self.nonlinearity = nn.Sigmoid()
        else:
            raise ValueError("<{}> is not a valid non-linearity function.".format(cfg["dnc_c"]["nonlinearity"]))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(cfg["hyperparameters"]["dropout"])
        if cfg["dnc_c"]["type"] == "MLP":
            self.dnc = DNC_MLP(
                input_size=cfg["hyperparameters"]["commun_embed_size"],
                output_size=cfg["dnc_c"]["output_size"],
                hidden_size=cfg["dnc_c"]["hidden_dim"],
                num_hidden_layers=cfg["dnc_c"]["num_layers_hidden"],
                dropout=cfg["dnc_c"]["dropout"],
                nr_cells=cfg["dnc_c"]["n"],
                cell_size=cfg["dnc_c"]["w"],
                read_heads=cfg["dnc_c"]["r"],
                nonlinearity=self.nonlinearity,
                gpu_id=cfg["hyperparameters"]["gpu_id"],
                debug=cfg["dnc_c"]["debug"],
                clip=20,
            )
        elif cfg["dnc_c"]["type"] == "LSTM":
            self.dnc = DNC(
                input_size=cfg["hyperparameters"]["commun_embed_size"],
                output_size=cfg["dnc_c"]["output_size"],
                hidden_size=cfg["dnc_c"]["hidden_dim"],
                rnn_type=cfg["dnc_c"]["rnn_type"],
                num_layers=cfg["dnc_c"]["num_layers"],
                num_hidden_layers=cfg["dnc_c"]["num_layers_hidden"],
                bias=True,
                batch_first=True,
                dropout=cfg["dnc_c"]["dropout"],
                bidirectional=cfg["dnc_c"]["bidirectional"],
                nr_cells=cfg["dnc_c"]["n"],
                cell_size=cfg["dnc_c"]["w"],
                read_heads=cfg["dnc_c"]["r"],
                gpu_id=cfg["hyperparameters"]["gpu_id"],
                independent_linears=True,
                share_memory=True,
                debug=cfg["dnc_c"]["debug"],
                clip=20)
        else:
            raise ValueError("dnc controller type <{}> is not defined".format(cfg["dnc"]["dnc_c_type"]))

        if cfg["dnc_c"]["concat_out_rv"]:
            in_fc_1 = cfg["dnc_c"]["output_size"] + cfg["dnc_c"]["w"] * cfg["dnc_c"]["r"]
        else:
            in_fc_1 = cfg["dnc_c"]["output_size"]
        self.fc_1 = nn.Linear(in_fc_1, cfg["hyperparameters"]["ans_vocab_size"])
        self.fc_2 = nn.Linear(cfg["hyperparameters"]["ans_vocab_size"], cfg["hyperparameters"]["ans_vocab_size"])

    def load_pretrained_weights(self, fc_flag):
        dnc_q_path = self.cfg["paths"]["dnc_q"]
        dnc_c_path = self.cfg["paths"]["dnc_c"]
        dnc_q_chkpt = torch.load(dnc_q_path)
        dnc_c_chkpt = torch.load(dnc_c_path)
        dnc_q = dnc_q_chkpt["net"]
        dnc_c = dnc_c_chkpt["net"]
        for n_pretrained, p_pretrained in dnc_q.qst_encoder.named_parameters():
            self.qst_encoder.state_dict()[n_pretrained].copy_(p_pretrained)

        for n_pretrained, p_pretrained in dnc_c.dnc.named_parameters():
            self.dnc.state_dict()[n_pretrained].copy_(p_pretrained)
        if fc_flag:
            self.state_dict()["fc_1.weight"].copy_(dnc_c.state_dict()["fc_1.weight"])
            self.state_dict()["fc_1.bias"].copy_(dnc_c.state_dict()["fc_1.bias"])
            self.state_dict()["fc_2.weight"].copy_(dnc_c.state_dict()["fc_2.weight"])
            self.state_dict()["fc_2.bias"].copy_(dnc_c.state_dict()["fc_2.bias"])

    def check_successul_weight_loading(self, fc_flag):
        dnc_q_path = self.cfg["paths"]["dnc_q"]
        dnc_c_path = self.cfg["paths"]["dnc_c"]
        dnc_q_chkpt = torch.load(dnc_q_path)
        dnc_c_chkpt = torch.load(dnc_c_path)
        dnc_q = dnc_q_chkpt["net"]
        dnc_c = dnc_c_chkpt["net"]
        state_dict_q = self.qst_encoder.state_dict()
        for n_pretrained, p_pretrained in dnc_q.qst_encoder.named_parameters():
            assert torch.equal(state_dict_q[n_pretrained], p_pretrained)

        state_dict_c = self.dnc.state_dict()
        for n_pretrained, p_pretrained in dnc_c.dnc.named_parameters():
            assert torch.equal(state_dict_c[n_pretrained], p_pretrained)
        if fc_flag:
            assert torch.equal(self.state_dict()["fc_1.weight"], dnc_c.state_dict()["fc_1.weight"])
            assert torch.equal(self.state_dict()["fc_1.bias"], dnc_c.state_dict()["fc_1.bias"])
            assert torch.equal(self.state_dict()["fc_2.weight"], dnc_c.state_dict()["fc_2.weight"])
            assert torch.equal(self.state_dict()["fc_2.bias"], dnc_c.state_dict()["fc_2.bias"])
        print("Weights successfully loaded!")

    def forward(self, img, qst, chx_q=None, mhx_q=None, rv_q=None, chx_c=None, mhx_c=None, rv_c=None):
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature, (chx_q, mhx_q, rv_q), v_q = self.qst_encoder(
            qst, chx=chx_q, mhx=mhx_q, rv=rv_q)

        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        if self.cfg["hyperparameters"]["share_memory"]:
            if mhx_c is not None:
                mhx_c["memory"] = mhx_q["memory"].detach().clone()
        v_c = None
        if self.cfg["dnc_c"]["type"] == "MLP":
            if self.cfg["dnc_c"]["debug"]:
                combined_feature, (mhx_c, rv_c), v_c = self.dnc(
                    combined_feature, mhx_c, rv_c)
            else:
                combined_feature, (mhx_c, rv_c) = self.dnc(
                    combined_feature, mhx_c, rv_c)

            if self.cfg["dnc_c"]["concat_out_rv"]:
                last_feat = torch.cat((combined_feature, rv_c), 1)
            else:
                last_feat = combined_feature
            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            last_feat = self.fc_1(last_feat)
            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            logits    = self.fc_2(last_feat)
            return logits, (chx_q, mhx_q, rv_q), v_q, (mhx_c, rv_c), v_c

        elif self.cfg["dnc_c"]["type"] == "LSTM":
            combined_feature = combined_feature.unsqueeze(1)
            if self.cfg["dnc_c"]["debug"]:
                combined_feature, (chx_c, mhx_c, rv_c), v_c = self.dnc(
                    combined_feature, (None, mhx_c, None))
            else:
                combined_feature, (chx_c, mhx_c, rv_c) = self.dnc(
                    combined_feature, (None, mhx_c, None))

            combined_feature = combined_feature.squeeze(1)
            if self.cfg["dnc_c"]["concat_out_rv"]:
                last_feat = torch.cat((combined_feature, rv_c), 1)
            else:
                last_feat = combined_feature

            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            last_feat = self.fc_1(last_feat)
            last_feat = self.nonlinearity(last_feat)
            last_feat = self.dropout(last_feat)
            logits    = self.fc_2(last_feat)
            return logits, (chx_q, mhx_q, rv_q), v_q, (chx_c, mhx_c, rv_c), v_c
