import torch.nn as nn


# Define Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 14)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 7)
        self.fc4 = nn.Linear(14, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(num_features=14)
        self.batchnorm2 = nn.BatchNorm1d(num_features=256)
        self.batchnorm2_2 = nn.BatchNorm1d(num_features=32)
        self.batchnorm3 = nn.BatchNorm1d(num_features=7)

    def forward(self, x):
        x = self.batchnorm1(self.relu(self.fc1(x)))
        # x = self.batchnorm2(self.relu(self.fc2(x)))
        # x = self.batchnorm2_2(self.relu(self.fc2_2(x)))
        # x = self.batchnorm3(self.relu(self.fc3(x)))
        # x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x




# Define Model
class Net_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm = nn.BatchNorm1d(num_features=10)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x



# Define Model
class Net_test2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 14)
        self.fc2 = nn.Linear(14, 7)
        self.fc3 = nn.Linear(7, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=14)
        self.batchnorm2 = nn.BatchNorm1d(num_features=7)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        # x = self.batchnorm1(x)
        x = self.sigmoid(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


# Define Model
class Net_test3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=40)
        self.batchnorm2 = nn.BatchNorm1d(num_features=10)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        # x = self.batchnorm1(x)
        x = self.sigmoid(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x
