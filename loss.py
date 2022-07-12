import torch
import random
import numpy as np
import torch.nn.functional as F


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_data():
    X = torch.from_numpy(np.load('test_data.npy')).type(torch.FloatTensor)
    Y = torch.from_numpy(np.load('test_label.npy')).type(torch.LongTensor)
    print(X.shape)  # shape: (frame_number, 24)
    print(Y.shape)  # shape: (frame_number)
    N, D_in, H, D_out = X.shape[0], X.shape[1], 512, 2
    return N, D_in, H, D_out, X, Y  # size, feature, hidden, class, data, label


def define_model(D_in, H, D_out, dropout):
    net = torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(D_in, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.Linear(H, H//2),
        torch.nn.BatchNorm1d(H//2),
        torch.nn.Linear(H//2, H // 4),
        torch.nn.BatchNorm1d(H // 4),
        torch.nn.Linear(H//4, H // 8),
        torch.nn.BatchNorm1d(H // 8),
        torch.nn.Linear(H // 8, D_out)
    )

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
            super(Net, self).__init__()
            self.dropout = torch.nn.Dropout(dropout)

            self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.bn1 = torch.nn.BatchNorm1d(n_hidden)

            self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden // 2)
            self.bn2 = torch.nn.BatchNorm1d(n_hidden // 2)

            self.hidden_3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)  # hidden layer
            self.bn3 = torch.nn.BatchNorm1d(n_hidden // 4)

            self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
            self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

            self.out = torch.nn.Linear(n_hidden // 8, n_output)  # output layer

        def forward(self, x):
            x = F.relu(self.hidden_1(x))  # activation function for hidden layer
            x = self.dropout(self.bn1(x))
            x = F.relu(self.hidden_2(x))  # activation function for hidden layer
            x = self.dropout(self.bn2(x))
            x = F.relu(self.hidden_3(x))  # activation function for hidden layer
            x = self.dropout(self.bn3(x))
            x = F.relu(self.hidden_4(x))  # activation function for hidden layer
            x = self.dropout(self.bn4(x))
            x = self.out(x)
            return x
    net = Net(D_in, H, D_out, dropout)
    return net


def define_loss():
    Loss = torch.nn.CrossEntropyLoss()
    return Loss


def define_optimizer():
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    return optimizer


def train(x, y, net, Loss, optimizer):
    for t in range(1000):
        y_pred = net(x)
        loss = Loss(y_pred, y)
        print("第{}次，损失为{}".format(t+1, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = net(x)
    print(y_pred.shape)
    loss = Loss(y_pred, y)
    print("训练完成，损失为{}".format(loss.item()))
    # save model
    net_path = 'net.pth'
    torch.save(net.state_dict(), net_path)

    return net_path


def test(x, y, net_path, Loss):
    net = define_model(24, 512, 2, 0.3)
    net.load_state_dict(torch.load(net_path))
    y_pred = net(x)
    loss = Loss(y_pred, y)
    print("测试完成，损失为{}".format(loss.item()))
    return 0


if __name__ == '__main__':
    N, D_in, H, D_out, X, Y = load_data()

    net = define_model(D_in, H, D_out, 0.3)
    Loss = define_loss()
    optimizer = define_optimizer()

    net_path = train(X, Y, net, Loss, optimizer)
    test(X, Y, net_path, Loss)
