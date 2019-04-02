# HackBU-Intro-to-ML

## Getting Started
Welcome to HackBU's Introduction to Machine Learning workshop! Today, we'll be covering some commonly used machine learning algorithms. To get started, make sure you have Python 3 (and optionally Git) installed. If you need help at any point, please ask one of the organizers for help!

## Part 1: Machine Learning with scikit-learn
### Part 1a: Finding and Exploring a Dataset
The first step to developing a good machine learning algorithm is using a good dataset. Many of the most accurate machine learning algorithms have millions if not billions of entries in their training data sets. Fortuntately for us, there are many robust datasets we can use to build our ML algorithms. 

The scikit-learn library comes with some good starting datasets. For today's activity, we'll be using the digits dataset, which contains images of handwritten numerical digits. To use this dataset, we'll import the load_digits function from sklearn.datasets and store it in a variable called digits.

```
from sklearn.datasets import load_digits
digits = load_digits()
```

To get a better sense of what we're working with, let's take a look at the attributes of `digits`. If we add the following line to our code:
```
print(dir(digits))
```
we can see that the digits dataset has 5 attributes - DESCR, data, images, target, and target_names. If we want to know even more about the dataset, we can add
```
print(digits.DESCR)
```
which prints out the description of the dataset.

For thoroughness, we can print the shape of the dataset with
```
print(digits.data.shape)
```
which shows that the dataset has 1797 rows and 64 columns.

We can also use the Python library matplotlib to display the images in this dataset. Add the following code:
```
import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[0]) 
plt.show() 
```

to your script and run it. You'll see an image like

## TODO: Add image

You can find other useful datasets in the [official scikit-learn documentation](https://scikit-learn.org/stable/datasets/index.html).

### Part 1b: Implementing a Machine Learning Algorithm
For now, we'll start off with two regression-based algorithms for supervised learning - Linear Regression and Logistic Regression.


We'll start by importing both algorithms from scikit-learn.
```
from sklearn.linear_model import LinearRegression, LogisticRegression
```

