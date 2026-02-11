import pytest
import numpy as np
from sklearn.dummy import DummyClassifier
from multiviewstacking import MultiViewStacking


def test_inconsistent_views():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)
    wrong_views = [[0, 1], [2, 3, 4], [5, 6]]  # invalid index

    with pytest.raises(Exception):
        MultiViewStacking(
            views_indices=wrong_views,
            first_level_learners=[DummyClassifier(), DummyClassifier()],
            meta_learner=DummyClassifier(),
        ).fit(X, y)


def test_predict_without_fit():
    model = MultiViewStacking(
        views_indices=[[0, 1], [2, 3]],
        first_level_learners=[DummyClassifier(), DummyClassifier()],
        meta_learner=DummyClassifier(),
    )

    X = np.random.rand(5, 4)
    with pytest.raises(Exception):
        model.predict(X)
