import numpy as np
import os
import torch
import random
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import time

import test
import loss
import plot
import GRU_net


def seed_():
    # seed
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # device = GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_load(n):   # n: Size
    # load data
    data_path = os.listdir('raw_npy/')
    data = []
    for i in range(0, n):
        x = np.load('raw_npy/' + data_path[i], allow_pickle=True).item()
        data_1 = torch.from_numpy(x.get('skel_body0').astype(np.float32)).unsqueeze(0)
        if x.get('nbodys')[0] == 2:     # finger out if there are two persons in a frame
            data_2 = torch.from_numpy(x.get('skel_body1').astype(np.float32)).unsqueeze(0)
        else:
            data_2 = torch.from_numpy(x.get('skel_body0').astype(np.float32)).unsqueeze(0)
        data.append(torch.cat((data_1, data_2), dim=0))
        print(data[i].shape)
    return data     # shape: (n, 2, frame_number, 25, 3)


def skeletons_plot(data_1, data_2):
    sk = plot.Draw3DSkeleton(data_1, data_2)
    # sk = Draw3DSkeleton(noise, noise)
    sk.visual_skeleton()


def gru_model(data):    # shape: (frame_number, 25, 3)
    in_ = data.transpose(0, 1)  # shape: (25, frame_number, 3)
    ou_ = in_
    input_dim = len(data.shape[0] * 25)
    output_dim = 3  # len(x.get('nbodys')*25)
    enc_emb_dim = 3
    dec_emb_dim = 3
    hid_dim = 10
    enc = GRU_net.Encoder(input_dim, enc_emb_dim, hid_dim)
    dec = GRU_net.Decoder(output_dim, dec_emb_dim, hid_dim)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = GRU_net.Seq2Seq(enc, dec, device).to(device)

    # Adam as the optimizer
    optimizer = optim.Adam(model.parameters())

    model.apply(GRU_net.init_weights)

    print(model(in_, ou_).transpose(0, 1).shape)
    return model(in_, ou_).transpose(0, 1)      # shape: (frame_number, 25, 3)


def fea_tran(raw_data):
    d = test.tran(raw_data)     # feature conversion
    return d                    # from (frame_number, 25, 3) to (frame_number, 24)


def human_model(data):
    net = loss.define_model(24, 512, 2, 0.3)
    net.load_state_dict(torch.load('net.pth'))  # defined model
    y_pred = net(data)
    y = torch.zeros(data.shape[0])  # the human model mark as 0
    Loss = loss.Loss(y_pred, y)     # CrossEntropyLoss
    print("测试完成，损失为{}".format(Loss.item()))
    return Loss


def train(Data):    # shape of data: (Size, 2, frame_number, 25, 3)
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0  # human_model loss
    for i in range(0, len(Data)):
        # anonymization
        if Data[i][0][0][0][0] == Data[i][1][0][0][0]:  # finger out if there are two persons in a frame
            Data_1 = gru_model(Data[i][0])  # shape of Data_1: (frame_number, 25, 3), tensor
            Data_2 = Data_1
        else:
            Data_1 = gru_model(Data[i][0])
            Data_2 = gru_model(Data[i][1])

        # human_model_loss
        fea_Data_1 = fea_tran(Data_1)
        fea_Data_2 = fea_tran(Data_2)
        loss_3 = loss_3 + (human_model(fea_Data_1) + human_model(fea_Data_2))/2

    whole_loss = (loss_1 + loss_2 + loss_3)/len(Data)

    return whole_loss


def model_test(Data):
    test_loss = 0

    return test_loss


def Ten_Fold(x):
    train_loss, test_loss = [], []

    # Shuffle Dateset with time.time()
    x = shuffle(x, random_state=round(time.time() * 1000000) % 1000)

    size = len(x)
    batch_size = size // 10  # just means 180/10, not actually "batch"
    index = 0  # For indexing the current test set
    sc = StandardScaler()

    for i in range(10):
        # Split Train and Test(Manually!)
        x_train = np.concatenate((x[0:index], x[index + batch_size:size]), axis=0)

        x_test = x[index:index + batch_size]

        train_loss.append(train(x_train))
        test_loss.append(model_test(x_test))

        # Update Index
        index = index + batch_size

    return train_loss, test_loss


if __name__ == '__main__':
    # seed
    seed_()
    # data
    Size = 100
    Data = data_load(Size)
    # plot
    ch = 0  # choose one skeleton data to plot
    skeletons_plot(Data[ch][0], Data[ch][1])
    # train
    Train_loss, Test_loss = Ten_Fold(Data)
