# Finding a Dataset
The first step to developing a good machine learning algorithm is using a good dataset. Many of the most accurate machine learning algorithms have millions if not billions of entries in their training data sets. Fortuntately for us, there are many small yet robust datasets we can use to build our ML algorithms. 

The scikit-learn library comes with some good starting datasets. For today's activity, we'll be recognizing handwritten numbers from scikit-learn's `digits` dataset. This dataset contains over 1700 labeled 8x8 pixel images of handrawn numerical digits.

To use this dataset, we'll import the `load_digits` function from `sklearn.datasets` and store it in a variable called `digits`.
```
from sklearn.datasets import load_digits
digits = load_digits()
```
&nbsp;

---

# Exploring a Dataset
To get a better sense of what we're working with, let's take a look at the attributes of `digits`. If we add the following line to our code, we can see that the digits dataset has 5 attributes - `DESCR`, `data`, `images`, `target`, and `target_names`. 
```
print(dir(digits))
```
&nbsp;

If we want to know even more about the dataset, we can print the description of `digits`.
```
print(digits.DESCR)
```
&nbsp;

For thoroughness, we can print the shape of the dataset with
```
print(digits.data.shape) # Should show 1797 rows and 64 columns
```
&nbsp;

We can also use the matplotlib library to display the images in this dataset. Add the following code to your script to display the first image in the dataset:
```
import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[0]) 
plt.show() 
```
&nbsp;

If all goes well, you will see the following image appear on your screen -
![matplotlib result](images/part1_matplotlib_image.png)
&nbsp;

### 📚 Further Reading
You can find other useful datasets in the [official scikit-learn documentation](https://scikit-learn.org/stable/datasets/index.html).

---

# Creating Training and Testing Sets


Now, we're going to split the data into two sets - a training set and a testing set. The training set will be used to train the machine learning algorithms, whereas the testing set will be used to verify the accuracy of the machine learning algorithms. 


To better visualize this relationship, think of a time where you studied for a math exam by completing practice problems, and tested your knowledge by completing the exam. The practice problems you completed were your training set, and the real exam was the testing set. 


⚠ **It is imperative that you keep your training and testing sets separate during the training process** - if your machine learning algorithm is tested with a data point it's already seen before, it may report a testing accuracy that is higher than it actually is.


Thankfully, scikit-learn gives us a method for automatically splitting up our full dataset into smaller training and testing sets.

```
from sklearn.model_selection import train_test_split
# random_state=42 seeds the random value with 42, meaning that everyone that runs this code will have the same accuracy.
# Machine learning algorithms have a degree of randomness to them, which can be mitigated by using the same random seed.
# Disregard this if you don't know what that means.
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.50, random_state=42)
```

In the above example, we import the `train_test_split` method from scikit-learn's `model_selection` sublibrary and use it to generate four smaller arrays:
* `X_train`, a two-dimensional array containing a certain amount of entries from the main dataset. Does not include the expected outcome of each data entry.
* `Y_train`, a one-dimensional array containing the expected outcome of each data entry in `X_train`.
* `X_test`, a two-dimensional array containing a certain amount of entries from the main dataset. Does not include the expected outcome of each data entry.
* `Y_test`, a one-dimensional array containing the expected outcome of each data entry in `X_test`.

Continuing our analogy of studying for a math exam, 
* `X_train` contains all your answers to the practice problems
* `Y_train` contains all the correct answers to the practice problems
* `X_test` contains all your answers to the real exam
* `Y_test` contains all the correct answers to the real exam


### 🤔 Food for Thought 
It can be tough to find a good ratio between the training and testing set size. In this case, we split it evenly (`test_size=0.5`), but many algorithms use much smaller testing set sizes (closer to 0.2). Although it may be tempting to improve your algorithm's accuracy by increasing the size of the training set, also consider that this will increase testing accuracy's margin of error.

---

# Implementing a Machine Learning Algorithm
Let's get to the fun part - implementing these algorithms.
For now, we'll start off with two regression-based algorithms for supervised learning - Linear Regression and Logistic Regression.
&nbsp;
&nbsp;

We'll start by importing both algorithms from scikit-learn.
```
from sklearn.linear_model import LinearRegression, LogisticRegression
```
**Linear Regression**
```
# Initialize a LinearRegression object
linear_model = LinearRegression()
# Fit the LinearRegression algorithm with the training data
linear_model.fit(X_train, Y_train)
```

**Logistic Regression**
```
# Initialize a LogisticRegression object
logistic_model = LogisticRegression()
# Fit the LogisticRegression algorithm with the training data
logistic_model.fit(X_train, Y_train)
```

And now to test these algorithms:
```
print("Linear Regression accuracy:", str(linear_model.score(X_test, Y_test) * 100) + "%")
print("Logistic Regression accuracy:", str(logistic_model.score(X_test, Y_test) * 100) + "%")
```

```
Linear Regression accuracy: 57.76594509083273%
Logistic Regression accuracy: 94.88320355951056%
```
&nbsp;

Clearly, logistic regression is a far more suitable algorithm for correctly determining a handwritten number - it achieves a 94.88% accuracy while linear regression is hardly better than a coinflip! But can we do better? 

**Answer:** Yes, with a neural network. 


### 📚 Further Reading
For an exhaustive list of the machine learning algorithms scikit-learn has to offer, check out [this page in their documentation](https://scikit-learn.org/stable/supervised_learning.html). Machine learning algorithms are not one size fits all - different problems require different algorithms. There are many cases where linear regression will outperform logistic regression, for instance, so it's good to understand the various types of machine learning algorithms.

