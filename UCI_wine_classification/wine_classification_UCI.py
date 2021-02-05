import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# turn off the 'not converge' warning
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

full_path = os.path.realpath(__file__)
cur_dir = os.path.dirname(full_path)

# Load dataset
url = os.path.join(cur_dir, "wine_class_UCI.csv")
attributes = ['class', 'Alcohol', 'Malic-acid', 'Ash', 'Alcalinity-of-ash ', 'Magnesium', 'Total-phenols',
'Flavanoids', 'Nonflavanoid-phenols', 'Proanthocyanins', 'Color-intensity', 'Hue', 'OD280/OD315-of-diluted-wines', 'Proline']
dataset = pandas.read_csv(url, names = attributes)

# convert class attribute from numerical to string
# since DataFrame.describe() only takes numerical
dataset['class'] = dataset['class'].apply(str)

#print(dataset)
print(dataset.describe())
print(dataset.groupby('class').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
# plt.show()

# histograms
# dataset[['Alcohol', 'Malic-acid', 'Ash', 'Alcalinity-of-ash ']].hist()
# plt.show()

# scatter plot of only three attributes, plot all will crash the computer by 13 * 13 subplots
# scatter_matrix(dataset[['Alcohol', 'Ash', 'Proanthocyanins']])
# plt.show()

# Split-out validation dataset
array = dataset.values
X = array[ : , 1:14]
Y = array[ : , 0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation =\
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle("Algorithm Comparision")
ax = fig.add_subplot(111)       # same as add_subplot(1, 1, 1)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print('The accuracy score is', accuracy_score(Y_validation, predictions))
print('====================================')
print('The confusion matrix is\n', confusion_matrix(Y_validation, predictions))
print('====================================')
print('The classification report is\n', classification_report(Y_validation, predictions))