# multiviewstacking: a python implementation of the Multi-View Stacking algorithm.

Description text.

The `multiviewstacking` package provides the following functionalities:

* Train Multi-View Stacking classifiers.
* Supports arbitrary number of views. The limit is your computer's memory.
* Use any scikit-learn classifier as first-level-learner and meta-learner.
* Use any custom model as long as they implement the `fit()`, `predict()`, and `predict_proba()` methods.
* Combine different types of first-level-learners.
* Comes with a pre-loaded dataset with two views for testing.


## Installation

You can install the `multiviewstacking` package with:

```
pip install multiviewstacking
```

## Quick start example

This quick start example shows you how to train a multi-view model. For more detailed tutorials, check the jupyter notebooks in the `examples` folder.

```{python}
import numpy as np
from data_utils import load_example_data
from multi_view_stacking import MultiViewStacking
from sklearn.ensemble import RandomForestClassifier

# Load the built-in example dataset.
(xtrain,ytrain,xtest,ytest,ind1,ind2,l) = load_example_data()
```

The built-in dataset contains features for two views (audio, accelerometer) for activity recognition.
The `load_example_data()` method returns a tuple with the train and test sets. It also returns the column indices for the two views and a LabelEnconder to convert the classes from integers back to strings.

```{python}
# Define two first-level-learners and the meta-learner.
# All of them are Random Forests but they can be any other model.
m_v1 = RandomForestClassifier(n_estimators=50, random_state=123)
m_v2 = RandomForestClassifier(n_estimators=50, random_state=123)
m_meta = RandomForestClassifier(n_estimators=50, random_state=123)

# Train the model.
model = MultiViewStacking(views_indices = [ind1, ind2],
                      first_level_learners = [m_v1, m_v2],
                      meta_learner = m_meta)
```

The `view_indices` parameter is a list of lists. Each list specifies the column indices of the train set for each view.
In this case `ind1` stores the indices of the audio features and `ind2` contains the indices of the accelerometer features.
Th `first_level_learners` parameter is a list of scikit-learn models or any other custom models. The `meta-learnr` specifies the model to be used as the meta-learner.

```{python}
# Train the model.
model.fit(xtrain, ytrain)

# Make predictions on the test set.
preds = model.predict(xtest)

# Compuet the accuracy.
np.sum(ytest == preds) / len(ytest)
```



## Citation

To cite this package use:

```{r}
Enrique Garcia-Ceja (2024). multiviewstacking: A python implementation of the Multi-View Stacking algorithm.
Python package https://github.com/enriquegit/multiviewstacking
```

BibTex entry for LaTeX:

```{r}
@Manual{MVS,
    title = {multiviewstacking: A python implementation of the Multi-View Stacking algorithm},
    author = {Enrique Garcia-Ceja},
    year = {2024},
    note = {Python package},
    url = {https://github.com/enriquegit/multiviewstacking}
}
```
