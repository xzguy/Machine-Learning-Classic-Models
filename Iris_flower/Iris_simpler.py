import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# turn off the 'not converge' warning
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

full_path = os.path.realpath(__file__)
cur_dir = os.path.dirname(full_path)

# Load dataset
url = os.path.join(cur_dir, "iris.csv")
attributes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names = attributes)

print(repr(dataset))
print(dataset.describe())
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2),
             sharex=False, sharey=False)
# plt.show()

# histograms
dataset.hist()
# plt.show()

# scatter plot matrix
scatter_matrix(dataset)
# plt.show()

# Split-out validation dataset
array = dataset.values
X = array[ : , 0:4]
Y = array[ : , 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation =\
    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Make predictions on validation dataset
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print('The accuracy of prediction is', accuracy_score(Y_validation, predictions))