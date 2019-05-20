import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return x


if __name__ == "__main__":
    num_classes = 20
    net = Net(num_classes)
    print(net)
