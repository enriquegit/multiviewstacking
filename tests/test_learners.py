import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from multiviewstacking import MultiViewStacking


# ---------------------------------------------------------------------
# Helper fixtures and dummy classes
# ---------------------------------------------------------------------

class DummyNoProba:
    """Model missing predict_proba."""
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(len(X))


class DummyNoFit:
    """Model missing fit method."""
    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


# ---------------------------------------------------------------------
# Tests for _validate_learners
# ---------------------------------------------------------------------

def test_validate_learners_passes_with_valid_models():
    """Should pass when all learners have fit() and predict_proba()."""
    model = MultiViewStacking(
        views_indices=[[0, 1], [2, 3]],
        first_level_learners=[RandomForestClassifier(), RandomForestClassifier()],
        meta_learner=RandomForestClassifier(),
        k=2
    )
    # Should not raise
    model._validate_learners()


def test_validate_learners_raises_for_missing_predict_proba():
    """Should raise AttributeError when a learner lacks predict_proba()."""
    with pytest.raises(AttributeError, match="predict_proba"):
        MultiViewStacking(
            views_indices=[[0, 1], [2, 3]],
            first_level_learners=[DummyNoProba(), RandomForestClassifier()],
            meta_learner=RandomForestClassifier(),
            k=2
        )


def test_validate_learners_raises_for_missing_fit():
    """Should raise TypeError when a learner lacks fit()."""
    with pytest.raises(TypeError, match="fit"):
        MultiViewStacking(
            views_indices=[[0, 1], [2, 3]],
            first_level_learners=[DummyNoFit(), RandomForestClassifier()],
            meta_learner=RandomForestClassifier(),
            k=2
        )
