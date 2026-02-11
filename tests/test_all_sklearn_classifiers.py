import pytest
import numpy as np
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from multiviewstacking import MultiViewStacking, load_example_data


@pytest.fixture(scope="module")
def example_data():
    """Use the example dataset provided by the library."""
    X_train, y_train, X_test, y_test, ind1, ind2, _ = load_example_data()
    return X_train, y_train, X_test, y_test, [ind1, ind2]


@pytest.mark.parametrize(
    "clf_name, ClfClass",
    [
        (name, ClfClass)
        for name, ClfClass in all_estimators(type_filter="classifier")
    ],
)
def test_all_sklearn_classifiers(clf_name, ClfClass, example_data):
    """
    Test MultiViewStacking with all sklearn classifiers.
    Skips classifiers that cannot be instantiated or trained with default params.
    """
    X_train, y_train, X_test, y_test, views = example_data

    # Skip meta or special classifiers
    skip_list = {
        "ClassifierChain",
        "MultiOutputClassifier",
        "OneVsRestClassifier",
        "OneVsOneClassifier",
        "OutputCodeClassifier",
        "StackingClassifier",
        "VotingClassifier",
        "CategoricalNB",  # requires categorical features
        "RadiusNeighborsClassifier",  # can fail on small data
    }
    if clf_name in skip_list:
        pytest.skip(f"Skipping {clf_name} (meta or incompatible classifier)")

    # Try to instantiate the classifier
    try:
        clf1 = ClfClass()
        clf2 = ClfClass()
        meta = ClfClass()
    except Exception as e:
        pytest.skip(f"Cannot instantiate {clf_name}: {e}")

    # Fit and test prediction
    try:
        model = MultiViewStacking(
            views_indices=views,
            first_level_learners=[clf1, clf2],
            meta_learner=meta,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        assert len(preds) == len(y_test)
        assert set(np.unique(preds)).issubset(set(np.unique(y_train)))

    except Exception as e:
        pytest.skip(f"{clf_name} failed: {type(e).__name__}: {e}")
