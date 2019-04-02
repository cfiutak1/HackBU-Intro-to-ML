# Improving on our Previous Algorithms with our First Neural Network
We were able to achieve a respectable accuracy on scikit-learn's digits dataset using logistic regression. However, the dataset qas quite small and simple - each image was only 8 pixels by 8 pixels. Many monitors today support a resolution of at least 1920 pixels by 1080 pixels, so 8x8 is almost primitive compared to real-world data we'd be working with.

Take, for instance, the MNIST dataset, which contains 28 x 28 pixel handwritten digits for classification. If we train a logistic regression model on this data, we'll see a noticeable decrease in performance. 
```
import tensorflow as tf

# Load in the MNIST dataset from TensorFlow
mnist = tf.keras.datasets.mnist

# Split the MNIST set into training and testing sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train / 255, X_test / 255

# Reshape the training and testing sets for compatibility with sklearn's LogisticRegression()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


from sklearn.linear_model import LogisticRegression

# Initialize a logisticRegression object
logistic_model = LogisticRegression()
# Fit the logisticRegression algorithm with the training data
logistic_model.fit(X_train[:10000, :], Y_train[:10000])

print("Logistic Regression Regression accuracy:", str(logistic_model.score(X_test, Y_test) * 100) + "%")
```

While logistic regression achieved ~95% accuracy on the digits dataset, it only achieves ~90% on the MNIST dataset.


