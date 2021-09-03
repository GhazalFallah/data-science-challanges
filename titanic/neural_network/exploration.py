import pandas as pd

train_set = pd.read_csv('../data/train.csv')
#
# train_set.head()
# print(f'You have {train_set.shape[0]} training samples with {train_set.shape[1]} features.')
# train_set.describe()
#
# y = train_set[['Survived']]
#
print(train_set.isnull().sum(0))
print(train_set.isnull().sum(1))
# x = train_set[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
# x = x.fillna(x[['Age']].median())
# y = y[x.isnull().sum(1) == 0]
# x = x.dropna()
#
# x = x.replace(['male', 'female'], ['0', '1'])
# x = x.replace(['S', 'C', 'Q'], ['0', '1', '2'])
# x[['Sex']] = pd.to_numeric(x['Sex'])
# x[['Embarked']] = pd.to_numeric(x['Embarked'])
#
# x = torch.from_numpy(x.values).float().cuda()
# y = torch.from_numpy(y.values).float().cuda()

print('finish')