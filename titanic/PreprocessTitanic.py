import pandas as pd
from sklearn import preprocessing

def preprocess_titanic_train(train_set_path):

    train_set = pd.read_csv(train_set_path)
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

    x = preprocessing.normalize(x, axis=0)
    y = y.to_numpy()

    return x, y



def preprocess_titanic_test(test_set_path):

    test_set = pd.read_csv(test_set_path)
    id = test_set[['PassengerId']]
    x = test_set[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

    dummies1 = pd.get_dummies(x.Pclass, prefix='Pclass')
    x = x.drop(['Pclass'], axis=1)
    x = dummies1.join(x)

    x = x.replace(['male', 'female'], ['0', '1'])

    x = x.fillna(x[['Age']].mean())

    x = x.fillna(x[['Fare']].mean())

    for i, item in enumerate(x['Cabin']):
        if pd.notna(item):
            x['Cabin'][i] = item[0]
        else:
            x['Cabin'][i] = 'X'
    dummies2 = pd.get_dummies(x.Cabin, prefix='Cabin')

    # x = x.fillna(x[['Embarked']].mode())
    dummies3 = pd.get_dummies(x.Embarked, prefix='Embarked')

    x = x.drop(['Cabin', 'Embarked'], axis=1)
    x = x.join(dummies2)
    x = x.join(dummies3)

    x = preprocessing.normalize(x, axis=0)
    id = id.to_numpy()

    return x, id


print('finish')