from torch.utils.data.dataset import Dataset
import pandas as pd
import torch
from sklearn import preprocessing

class house_prices_train_set(Dataset):
    def __init__(self, dataset_path):
        train_set = pd.read_csv(dataset_path)
        y = train_set[['Survived']]
        x = train_set[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        x = x.fillna(x[['Age']].median())
        y = y[x.isnull().sum(1) == 0]
        x = x.dropna()
        x = x.replace(['male', 'female'], ['1', '2'])
        x = x.replace(['S', 'C', 'Q'], ['1', '3', '2'])
        x[['Sex']] = pd.to_numeric(x['Sex'])
        x[['Embarked']] = pd.to_numeric(x['Embarked'])
        # Preprocessing
        x = preprocessing.normalize(x,axis=0)
        # x = torch.from_numpy(x.values).float()
        # y = torch.from_numpy(y.values).float()
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y.values).float()
        # self.x = torch.squeeze(x)
        # self.y = torch.squeeze(y)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = (self.x[idx,:], self.y[idx])
        return sample