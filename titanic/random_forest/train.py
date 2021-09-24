from titanic.PreprocessTitanic import preprocess_titanic_train, preprocess_titanic_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std

# train
train_set_path = '../data/train.csv'
x, y = preprocess_titanic_train(train_set_path)
clf = RandomForestClassifier()
clf.fit(x, y)


# evaluate
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(clf, x, y.ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# predict
test_set_path = '../data/test.csv'
x, id = preprocess_titanic_test(test_set_path)
yhat = clf.predict(x)

print('finished')