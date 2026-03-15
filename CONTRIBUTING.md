# Contributing to multiviewstacking

Thank you for your interest in contributing to **multiviewstacking**!
This project welcomes contributions from the community. In particular, we are interested in contributions that extend the library to support **regression models** in addition to the current classification functionality.

---

## 1. Ways to Contribute

You can contribute in several ways:

* Implement **regression multi-view stacking** compatible with the multiview stacking framework
* Improve documentation
* Add tests or benchmarks
* Fix bugs or improve performance


---

# 2. Development Setup

First, fork the repository and clone it locally:

```bash
git clone https://github.com/enriquegit/multiviewstacking.git
cd multiviewstacking
```

Create a development environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

---

# 3. Branching Workflow

Please create a feature branch before starting work. For example:

```bash
git checkout -b feature/regression-model
```

---

# 4. Design Guidelines for Regression Models

The regression implementation should follow **scikit-learn conventions**.

### Required Base Classes

Regression models should inherit from:

```python
BaseEstimator
RegressorMixin
```

### Required Methods

Your estimator should implement:

```
fit()
predict()
```

### Input Format

The input data and parameters should follow the same formats as the ```MultiViewStacking``` class and with similar validations.

---

# 5. Code Quality Requirements

All contributions must follow these guidelines:

### Style

* Follow **PEP8**
* Use **type hints when possible**
* Document public methods

### Documentation

Every new class must include:

* A docstring describing the algorithm
* Parameter descriptions
* Usage example

Example:

```python
"""
Multiview Stacking Regressor.

This estimator combines predictions from multiple view-specific
regressors using a meta-regression model.

Parameters
----------
base_learners : list
    List of regressors for each view.

meta_learner : estimator
    Regression model used to combine predictions.
"""
```

---

# 6. Testing Requirements

All new functionality must include **unit tests**.

Tests should cover:

* Regression fitting
* Prediction shape
* Compatibility with multi-view input
* Use the included HTAD dataset (for classification).

Run tests with:

```bash
pytest
```

If you add regression support, include tests such as:

```
tests/test_regressor.py
```

---

# 7. Continuous Integration

Before submitting a pull request, ensure:

```
pytest
```

runs successfully and all tests pass.

---

# 8. Submitting a Pull Request

When your implementation is ready:

1. Push your branch:

```bash
git push origin feature/regression-model
```

2. Open a **Pull Request** on GitHub.

3. In the PR description include:

* Summary of the regression method implemented
* Description of the design
* Added tests
* Example usage

---


# 9. Questions

If you have questions about the architecture or implementation details, please open an **issue** before starting major changes.

---

Thank you for helping improve **multiviewstacking** and making multi-view machine learning tools more accessible to the community.
