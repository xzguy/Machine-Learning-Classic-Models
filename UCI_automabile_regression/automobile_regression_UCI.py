import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
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

# Define the headers since the data does not have any
attributes = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

access_data_online = True
if access_data_online:
    # Read in the CSV file and convert "?" to NaN
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
                        header=None, names=attributes, na_values="?" )
else:
    full_path = os.path.realpath(__file__)
    cur_dir = os.path.dirname(full_path)

    # Load dataset
    url = os.path.join(cur_dir, "automobile_regr_UCI.csv")
    # Read in the CSV file and convert "?" to NaN
    df = pd.read_csv(url, header=None, names=attributes, na_values="?" )

print(df.describe())
print(df.dtypes)

### extract object values ###
obj_df = df.select_dtypes(include=['object']).copy()
# clean NaN values, fill two missing 'num_doors' as 'four'
#print(obj_df[obj_df.isnull().any(axis=1)])
#print(obj_df["num_doors"].value_counts())
obj_df = obj_df.fillna({"num_doors": "four"})
# convert strings into numbers directly
cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                    "two": 2, "twelve": 12, "three": 3 }}
obj_df.replace(cleanup_nums, inplace=True)

obj_df = pd.get_dummies(obj_df, columns=['make', 'fuel_type', 'aspiration',\
'body_style', 'drive_wheels', 'engine_location', 'fuel_system'])

# lossy conversion of 'engine_type'
#print(obj_df["engine_type"].value_counts())
obj_df["engine_type"] = np.where(obj_df["engine_type"].str.contains("ohc"), 1, 0)

num_df = pd.concat([df.select_dtypes(exclude=['object']), obj_df], axis=1)
# delete data with NaN
num_df.dropna(axis='index', how='any', inplace=True)

print(num_df.describe())
num_attr = num_df.shape[1]

# Split-out validation data set
array = num_df.values
X = array[:, 1:num_attr]
Y = array[:, 0]       # first attribute 'symboling' is the class we choose, it can be others
validation_size = 0.2
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
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print('The accuracy score is', accuracy_score(Y_validation, predictions))
print('====================================')
print('The confusion matrix is\n', confusion_matrix(Y_validation, predictions))
print('====================================')
print('The classification report is\n', classification_report(Y_validation, predictions))
