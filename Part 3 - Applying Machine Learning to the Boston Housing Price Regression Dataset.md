# The Boston House-Price Dataset
The Boston house-price dataset is one of several datasets included with `sklearn`. It contains 506 samples of houses in the Boston area, with 13 different attributes measured of each (e.g. per capita crime, tax rate, pupil-teacher ratio, etc.), with the 'target' variable being the price of the house.

Accessing the data in the Boston house-price dataset is exactly the same as accessing the `digits` dataset.

```python
from sklearn.datasets import load_boston
...<more imports/code/whatever>...
boston = load_boston()

xtrain, xtest, ytrain, ytest = train_test_split(boston.data,
                                                boston.target,
                                                test_size=0.5,    # Tweak to your liking
                                                random_state=42)  # Set random seed
```

Let's see if we can come up with a model that will accurately predict the price of a house given its attributes.

# Models At Our Disposal
`sklearn` comes with a massive variety of models, but they all excel at different tasks. For example, convolutional neural networks are exceptionally good at classifying objects in images, where recurrent neural networks are great at things like generating text that follows grammar rules.

There is one large distinction between many models to make. There are *classifiers*, and *regressions*. 

 - __Classifiers *label* things, e.g.:__
    - This is picture of a cat
    - This data reminds me of an orange
    - This sounds like this person's voice
 - __Regressions *estimate* things, e.g.:__
    - This was probably a 3.5 on the Richter scale
    - This stock will grow 5% by tomorrow
    - *This house probably cost $100k

__Classifiers pick from a list of label options to predict what something is (i.e. apples, oranges), while regressions guess a value on a continuous spectrum (i.e. a number on the Richter scale, *the price of a house*)__

So, for this dataset, it will be most valid to use a *regression* to predict the prices of the test-case houses.

Thankfully, `sklearn` comes with many kinds of regression to help achieve this, like `LinearRegression` (sound familiar?), `Lasso`, `Ridge`, and many more, which can be found at the [official documentation](https://scikit-learn.org/stable/supervised_learning.html).

Also fortunately, many of them work the same! Here's the basic form:
```python
from sklearn.<some_model_group> import <ModelOfChoice>
...
# Create the model
model = <ModelOfChoice>()

# Train it
model.train(xtrain, ytrain)

# Check the accuracy (if that is your intention)
print("<ModelOfChoice> Accuracy:", str(model.score(xtest, ytest) * 100) + "%")
```
