import pandas as pd
from sklearn import preprocessing

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


y = train_set[['Survived']]
x = train_set[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

dummies1 = pd.get_dummies(x.Pclass, prefix='Pclass')
x = x.drop(['Pclass'], axis=1)
x = dummies1.join(x)

x = x.replace(['male', 'female'], ['0', '1'])

x = x.fillna(x[['Age']].mean())

for i, item in enumerate(x['Cabin']):
    if pd.notna(item):
        x['Cabin'][i] = item[0]
    else:
        x['Cabin'][i] = 'X'
dummies2 = pd.get_dummies(x.Cabin, prefix='Cabin')

x = x.fillna(x[['Embarked']].mode())
dummies3 = pd.get_dummies(x.Embarked, prefix='Embarked')

x = x.drop(['Cabin', 'Embarked'], axis=1)
x = x.join(dummies2)
x = x.join(dummies3)

# x.hist(column='Age')
# x['Age'].median()
# x['Age'].mean()
# x['Age'].mode()
# train_set['Embarked'].value_counts()

x = preprocessing.normalize(x,axis=0)
y = y.to_numpy()



print('finish')