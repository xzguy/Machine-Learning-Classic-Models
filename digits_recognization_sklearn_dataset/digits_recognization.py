import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# turn off the 'not converge' warning
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print('Image Data Shape' , digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

index = 23
num_pred = logisticRegr.predict(x_test[index].reshape(1,-1))
num_actu = y_test[index]
plt.subplot(111)
plt.imshow(np.reshape(x_test[index], (8,8)), cmap=plt.cm.gray)
plt.title('Actual: %i, Predict: %i\n' % (num_actu, num_pred), fontsize = 15)
plt.show()

predictions = logisticRegr.predict(x_test)

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

score = logisticRegr.score(x_test, y_test)
print(score)

### seaborn
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()