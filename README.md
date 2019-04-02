# HackBU-Intro-to-ML

## Getting Started
Welcome to HackBU's Introduction to Machine Learning workshop! Today, we'll be covering some commonly used machine learning algorithms. To get started, make sure you have Python 3 (and optionally Git) installed. If you need help at any point, please ask one of the organizers for help!

## Part 1: Machine Learning with scikit-learn
The first step to developing a good machine learning algorithm is using a good dataset. Many of the most accurate machine learning algorithms have millions if not billions of entries in their training data sets. Fortuntately for us, there are many robust datasets we can use to build our ML algorithms. 

The scikit-learn library comes with some good starting datasets. For today's activity, we'll be using the digits dataset, which contains images of handwritten numerical digits. To use this dataset, we'll import the load_digits function from sklearn.datasets and store it in a variable called digits.

`
from sklearn.datasets import load_digits
digits = load_digits()
`
