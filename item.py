import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = os.listdir('raw_npy/')
x = np.load('raw_npy/' + data_path[10], allow_pickle=True).item()
print(torch.from_numpy(x.get('skel_body0').astype(np.float32)).shape)


class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super(Encoder, self).__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        outputs, hidden = self.rnn(src)
        return hidden.transpose(0, 1)  # [batch size, 1, hid dim]


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        Hidden = hidden.expand(input.shape[0], input.shape[1], self.hid_dim)
        emb_con = torch.cat((input, Hidden), dim=2)
        # emb_con = [batch size, 25, emb dim + hid dim]
        output, hidden = self.rnn(emb_con)
        # output = [batch size, 25, hid dim]
        prediction = self.out(output)
        # prediction = [batch size, 25, emb dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg):
        # src = [batch size, 25, 3]
        # trg = [batch size, 25, 3]
        context = self.encoder(src)
        output, hidden = self.decoder(trg, context)
        return output


# 始化模型中的参数权重
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


# 该模型优化在于尝试encoder、decoder中的网络，当前按最简单层GRU与一层全连接层组合
if __name__ == '__main__':
    in_ = torch.from_numpy(x.get('skel_body0').astype(np.float32))  # 25为序列长度，3为序列特征维度
    ou_ = torch.from_numpy(x.get('skel_body0').astype(np.float32))
    OUTPUT_DIM = 3  # len(x.get('nbodys')*25)
    ENC_EMB_DIM = 3
    DEC_EMB_DIM = 3
    HID_DIM = 10
    enc = Encoder(ENC_EMB_DIM, HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = Seq2Seq(enc, dec, device).to(device)

    # Adam作为优化器
    optimizer = optim.Adam(model.parameters())

    model.apply(init_weights)

    print(model(in_, ou_).shape)
