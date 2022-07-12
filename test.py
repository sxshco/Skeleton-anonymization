import os
import numpy as np
import torch
import random
import math

# seed
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
# device = GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load raw data
data_path = os.listdir('raw_npy/')
x = np.load('raw_npy/' + data_path[50], allow_pickle=True).item()


# add noise
def noise_add(raw_data):
    y = np.zeros((raw_data.shape[0], 25, 3))
    for it in range(0, raw_data.shape[0]):
        for jt in range(0, 25):
            for kt in range(0, 3):
                y[it][jt][kt] = raw_data[it][jt][kt] + raw_data[it][jt][kt] * np.random.normal(0, 1)
    return y


# compute the length between the joint
def length(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2))


# compute the ratio of the length of joint
def data_normal(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    dst = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(dst)
    return norm_data


# transform the raw data for xyz coord to ratio of the length of joint
def tran(data):
    Data = torch.zeros(data.shape[0], 24)
    for i in range(0, data.shape[0]):
        Data[i][0] = length(data[i][0], data[i][1])
        Data[i][1] = length(data[i][1], data[i][20])
        Data[i][2] = length(data[i][20], data[i][2])
        Data[i][3] = length(data[i][2], data[i][3])
        Data[i][4] = length(data[i][0], data[i][16])
        Data[i][5] = length(data[i][0], data[i][12])
        Data[i][6] = length(data[i][20], data[i][8])
        Data[i][7] = length(data[i][20], data[i][4])
        Data[i][8] = length(data[i][16], data[i][17])
        Data[i][9] = length(data[i][12], data[i][13])
        Data[i][10] = length(data[i][17], data[i][18])
        Data[i][11] = length(data[i][13], data[i][14])
        Data[i][12] = length(data[i][18], data[i][19])
        Data[i][13] = length(data[i][14], data[i][15])
        Data[i][14] = length(data[i][8], data[i][9])
        Data[i][15] = length(data[i][4], data[i][5])
        Data[i][16] = length(data[i][9], data[i][10])
        Data[i][17] = length(data[i][5], data[i][6])
        Data[i][18] = length(data[i][10], data[i][11])
        Data[i][19] = length(data[i][6], data[i][7])
        Data[i][20] = length(data[i][11], data[i][24])
        Data[i][21] = length(data[i][7], data[i][22])
        Data[i][22] = length(data[i][11], data[i][23])
        Data[i][23] = length(data[i][7], data[i][21])
        Data[i] = data_normal(Data[i])

    return Data


def data_processing(data, noise):
    d = tran(data)
    n = tran(noise)
    d_label = torch.zeros(d.shape[0])  # mark as 0
    n_label = torch.ones(n.shape[0])  # mark as 1
    a = torch.cat((d, n), dim=0)  # (frame number*2, 24)
    b = torch.cat((d_label, n_label), dim=0)  # (frame number*2)
    index = [i for i in range(len(a))]
    np.random.shuffle(index)
    a = a[index]
    b = b[index]
    return a, b


if __name__ == '__main__':
    raw_data = torch.from_numpy(x.get('skel_body0').astype(np.float32))
    noise = noise_add(raw_data)
    np.save('noise', noise)
    test_data, test_label = data_processing(raw_data, noise)
    print(test_data.shape)
    print(test_label.shape)
    print(test_data.dtype)
    print(test_label.dtype)
    np.save('test_data', test_data)
    np.save('test_label', test_label)
