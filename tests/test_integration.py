import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from multiviewstacking import MultiViewStacking


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def binary_dataset():
    """Small binary classification dataset."""
    X = np.random.rand(60, 6)
    y = np.random.randint(0, 2, size=60)
    return X, y


@pytest.fixture
def multiclass_dataset():
    """Small multiclass classification dataset."""
    X = np.random.rand(90, 6)
    y = np.random.randint(0, 3, size=90)
    return X, y


def make_model():
    """Create a base MultiViewStacking instance."""
    return MultiViewStacking(
        views_indices=[[0, 1, 2], [3, 4, 5]],
        first_level_learners=[RandomForestClassifier(random_state=0),
                              LogisticRegression(max_iter=1000)],
        meta_learner=RandomForestClassifier(random_state=0),
        k=3,
        random_state=0
    )


# ---------------------------------------------------------------------
# Integration tests: fit, predict, predict_proba
# ---------------------------------------------------------------------

@pytest.mark.parametrize("dataset_fixture", ["binary_dataset", "multiclass_dataset"])
def test_fit_predict_predict_proba_end_to_end(dataset_fixture, request):
    """Full pipeline test for both binary and multiclass setups."""
    X, y = request.getfixturevalue(dataset_fixture)
    model = make_model()

    fitted = model.fit(X, y)

    # Check predictions
    preds = fitted.predict(X)
    probs = fitted.predict_proba(X)

    # Type and shape assertions
    assert isinstance(preds, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert probs.shape[0] == X.shape[0]
    assert len(preds) == len(y)

    # Probabilities should sum to 1 for each instance
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_fit_with_dataframe_input(binary_dataset):
    """Ensure that DataFrame inputs are accepted and yield correct outputs."""
    X, y = binary_dataset
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = make_model()

    fitted = model.fit(X_df, y)
    preds = fitted.predict(X_df)
    probs = fitted.predict_proba(X_df)

    assert len(preds) == len(y)
    assert probs.shape[1] in [2, len(np.unique(y))]


def test_predict_before_fit_raises(binary_dataset):
    """predict() or predict_proba() before fit() should raise NotFittedError."""
    X, _ = binary_dataset
    model = make_model()

    with pytest.raises(Exception):
        model.predict(X)

    with pytest.raises(Exception):
        model.predict_proba(X)
