import pytest
import numpy as np
from sklearn.dummy import DummyClassifier

from multiviewstacking import load_example_data, MultiViewStacking


@pytest.fixture
def example_data():
    X_train, y_train, X_test, y_test, ind1, ind2, _ = load_example_data()
    return X_train, y_train, X_test, y_test, [ind1, ind2]


def test_fit_and_predict(example_data):
    X_train, y_train, X_test, y_test, views = example_data
    model = MultiViewStacking(
        views_indices=views,
        first_level_learners=[DummyClassifier(strategy="most_frequent"), DummyClassifier()],
        meta_learner=DummyClassifier(),
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Check output shape
    assert preds.shape == y_test.shape
    # Check valid class labels
    assert set(np.unique(preds)).issubset(set(np.unique(y_train)))


def test_predict_proba(example_data):
    X_train, y_train, X_test, _, views = example_data
    model = MultiViewStacking(
        views_indices=views,
        first_level_learners=[DummyClassifier(), DummyClassifier()],
        meta_learner=DummyClassifier(strategy="uniform"),
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)

    # Check probability matrix shape and normalization
    assert probs.ndim == 2
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
