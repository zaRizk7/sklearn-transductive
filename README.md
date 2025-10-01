# `scikit-learn` Extension for Transductive Learning

Utilities that extend scikit-learn with a focus on transductive learning and domain adaptation workflows.

The project mirrors the public scikit-learn API whenever possible so that your existing estimators, pipelines, and model selection routines keep working. Extra utilities such as an Optuna-backed search CV, harmonization helpers, and targeted cross-validation strategies are layered on top to make domain adaptation experiments easier to run.

## Installation

### Prerequisites
- Python 3.9 or later
- `pip` (from Python or a virtual environment)

### Install the released package from GitHub

Install directly from the repository using `pip`:

```bash
pip install "git+https://github.com/zaRizk7/sklearn-transductive.git"
```

This command vendors the package into your current environment just like a regular PyPI release. If you are pinning dependencies elsewhere (e.g. `requirements.txt`, `pyproject.toml`), add the same `git+https://...` spec there as well.

### Editable install for local development

If you plan to contribute, install in editable mode after cloning:

```bash
git clone https://github.com/zaRizk7/sklearn-transductive.git
cd sklearn-transductive
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Quick start example

Below is a minimal example that targets evaluation on a specific domain while fitting on all available data. The snippet uses `TargetedGroupSplit` to ensure the target domain remains in the test fold during cross-validation.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn_transductive.model_selection import TargetedGroupSplit
from sklearn_transductive.pipeline import TransductivePipeline

# Synthetic example with two domains ("source" vs "target")
X, y = make_classification(n_samples=200, n_features=20, random_state=42)
groups = np.where(np.arange(len(X)) < 140, "source", "target")

# Wrap your estimator; TransductivePipeline keeps sklearn semantics
model = TransductivePipeline([("clf", LogisticRegression(max_iter=1_000))])

splitter = TargetedGroupSplit(cv=3, target_group="target")
scores = []
for train_idx, test_idx in splitter.split(X, y, groups):
    model.fit(X[train_idx], y[train_idx])
    scores.append(model.score(X[test_idx], y[test_idx]))

print(f"Average score on target domain: {np.mean(scores):.3f}")
```

The same splitter can wrap other scikit-learn CV iterators such as `StratifiedKFold` or `GroupKFold`. Leave `target_group=None` to fall back to standard cross-validation on the full dataset.

## Key features
- `TargetedGroupSplit` for focused evaluation on a single domain while keeping the rest of the data available for fitting.
- Compatibility layers that wrap scikit-learn models and pipelines so you can drop transductive components into existing projects.
- Optional integration with Optuna to power Bayesian hyperparameter search (`OptunaSearchCV`).

## Getting help
- Open an issue on GitHub if something is unclear or broken.
- Contributions are welcome—feel free to submit pull requests with enhancements, documentation, or bug fixes.
