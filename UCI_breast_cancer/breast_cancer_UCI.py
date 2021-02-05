import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

full_path = os.path.realpath(__file__)
cur_dir = os.path.dirname(full_path)

# Load dataset
url = os.path.join(cur_dir, "breast_cancer_UCI.csv")
attributes = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size',
'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size',
'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
dataset = pd.read_csv(url, names=attributes, na_values="?" )

# convert class attribute from numerical to string
# since DataFrame.describe() only takes numerical
cleanup_nums = {"Class":     {2: "benign", 4: "malignant"}}
dataset.replace(cleanup_nums, inplace=True)
# delete data with NaN
dataset.dropna(axis='index', how='any', inplace=True)

#print(dataset.head(5))
print(dataset.describe())
print(dataset.groupby('Class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,6), sharex=False, sharey=False)
plt.show()

# Split-out validation dataset
array = dataset.values
# ignore the ID is better for most models
X = array[ : , 1:10]
Y = array[ : , 10]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation =\
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
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
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('The accuracy score is', accuracy_score(Y_validation, predictions))
print('====================================')
print('The confusion matrix is\n', confusion_matrix(Y_validation, predictions))
print('====================================')
print('The classification report is\n', classification_report(Y_validation, predictions))