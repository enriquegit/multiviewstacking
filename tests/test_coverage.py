import pytest
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from multiviewstacking import MultiViewStacking, load_example_data

# ---------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------

def test_fit_and_predict_basic():
    """Test fitting and predicting using two simple views."""
    X_train, y_train, X_test, y_test, ind_v1, ind_v2, _ = load_example_data()

    learners = [GaussianNB(), GaussianNB()]
    meta = RandomForestClassifier(n_estimators=10, random_state=123)

    model = MultiViewStacking(
        views_indices=[ind_v1, ind_v2],
        first_level_learners=learners,
        meta_learner=meta,
        k=5,
        random_state=123
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    acc = np.mean(preds == y_test)
    assert 0.5 < acc <= 1.0  # reasonable range


def test_score_functionality():
    """Ensure that the score() method returns a float accuracy value."""
    X_train, y_train, X_test, y_test, ind_v1, ind_v2, _ = load_example_data()

    model = MultiViewStacking(
        views_indices=[ind_v1, ind_v2],
        first_level_learners=[GaussianNB(), GaussianNB()],
        meta_learner=LogisticRegression(max_iter=200),
        k=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert isinstance(score, float)
    assert 0 <= score <= 1


# ---------------------------------------------------------------------
# Validation and internal consistency
# ---------------------------------------------------------------------


def test_validate_learners_consistency():
    """Test internal learner validation with mismatched views."""
    X_train, y_train, *_ = load_example_data()

    # only one learner for two views should raise ValueError
    with pytest.raises(ValueError):
        MultiViewStacking(
            views_indices=[[0, 1, 2], [3, 4, 5]],
            first_level_learners=[GaussianNB()],
        )


def test_reproducibility_with_random_state():
    """Ensure that random_state controls reproducibility."""
    X_train, y_train, X_test, y_test, ind_v1, ind_v2, _ = load_example_data()

    model1 = MultiViewStacking(
        views_indices=[ind_v1, ind_v2],
        first_level_learners=[GaussianNB(), GaussianNB()],
        random_state=123,
    )
    model2 = MultiViewStacking(
        views_indices=[ind_v1, ind_v2],
        first_level_learners=[GaussianNB(), GaussianNB()],
        random_state=123,
    )

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    preds1 = model1.predict(X_test)
    preds2 = model2.predict(X_test)
    assert np.all(preds1 == preds2)


# ---------------------------------------------------------------------
# Edge and error handling
# ---------------------------------------------------------------------

def test_incorrect_input_shapes():
    """Ensure informative errors for incorrect input shapes."""
    X_train, y_train, *_ = load_example_data()
    model = MultiViewStacking(first_level_learners=[GaussianNB()])
    with pytest.raises(ValueError):
        model.fit(X_train.iloc[:, :5], y_train)  # missing view definitions


def test_custom_meta_learner_with_proba():
    """Ensure predict_proba works when meta-learner supports it."""
    X_train, y_train, X_test, _, ind_v1, ind_v2, _ = load_example_data()

    model = MultiViewStacking(
        views_indices=[ind_v1, ind_v2],
        first_level_learners=[GaussianNB(), GaussianNB()],
        meta_learner=RandomForestClassifier(n_estimators=5, random_state=1),
        k=3,
        random_state=1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    assert isinstance(probs, np.ndarray)
    assert probs.shape[0] == X_test.shape[0]

# ---------------------------------------------------------------------
# Smoke test for __repr__ and attributes
# ---------------------------------------------------------------------

def test_model_summary_and_attributes():
    """Ensure the print and attributes of the model work correctly."""
    X_train, y_train, *_ = load_example_data()
    model = MultiViewStacking(
        first_level_learners=[GaussianNB()],
        views_indices=[[i for i in range(10)]]
    )
    text = str(model)
    assert "MultiViewStacking" in text
    assert hasattr(model, "k")
    assert hasattr(model, "random_state")
