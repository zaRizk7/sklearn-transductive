# `scikit-learn` Extension for Transductive Learning

Utilities that extend scikit-learn with a focus on transductive learning and domain adaptation workflows.

The project mirrors the public scikit-learn API whenever possible so that your existing estimators, pipelines, and model selection routines keep working. Extra utilities such as an Optuna-backed search CV, harmonization helpers, and targeted cross-validation strategies are layered on top to make domain adaptation experiments easier to run.

## Installation

### Prerequisites
- Python 3.10 or later
- `pip` (from Python or a virtual environment)

### Install the released package from GitHub

Install directly from the repository using `pip`:

```bash
pip install "git+https://github.com/zaRizk7/sklearn-transductive.git"
```

This command vendors the package into your current environment just like a regular PyPI release. If you are pinning dependencies elsewhere (e.g. `requirements.txt`, `pyproject.toml`), add the same `git+https://...` spec there as well.

### Editable install for local development

If you plan to modify the code, install in editable mode after cloning:

```bash
git clone https://github.com/zaRizk7/sklearn-transductive.git
cd sklearn-transductive
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Quick start example

The snippet below adapts features with `MIDA`, delegates supervised learning to `SupervisedOnlyEstimator`, and composes the split extensions so that target-domain samples stay unlabeled during fitting but remain the focus of the evaluation.

```python
from imblearn.pipeline import Pipeline
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn_transductive.adaptation import MIDA
from sklearn_transductive.model_selection import (
    SemiSupervisedSplit,
    SplitComposer,
    TargetedGroupSplit,
    TransductiveSplit,
)
from sklearn_transductive.pipeline import SupervisedOnlyEstimator

sklearn.set_config(enable_metadata_routing=True)

# Synthetic dataset with a shifted target domain
rng = np.random.default_rng(42)
X, y_true = make_classification(
    n_samples=400,
    n_features=20,
    n_informative=12,
    random_state=0,
)
domains = np.where(np.arange(len(X)) < 300, "source", "target")
X[domains == "target"] += rng.normal(loc=1.0, scale=0.3, size=X[domains == "target"].shape)

# Only some target samples are labeled; the rest stay transductive
y_observed = y_true.copy()
target_idx = np.flatnonzero(domains == "target")
rng.shuffle(target_idx)
unlabeled_target = target_idx[40:]
y_observed[unlabeled_target] = -1

transductive_model = Pipeline(
    steps=[
        ("adapt", MIDA()),
        ("clf", SupervisedOnlyEstimator(LogisticRegression())),
    ]
)

splitter = SplitComposer(
    cv=KFold(n_splits=3, shuffle=True, random_state=0),
    steps=[
        TransductiveSplit(),
        SemiSupervisedSplit(),
        TargetedGroupSplit(target_group="target"),
    ],
)

scores = []
for train_idx, test_idx in splitter.split(X, y_observed, domains):
    y_train = y_observed[train_idx].copy()
    # Drop labels for the evaluation fold so they remain transductive during fit
    y_train[np.isin(train_idx, test_idx)] = -1

    transductive_model.fit(X[train_idx], y_train, domains=domains[train_idx])
    y_pred = transductive_model.predict(X[test_idx], domains=domains[test_idx])
    scores.append(accuracy_score(y_true[test_idx], y_pred))

print(f"Average target accuracy: {np.mean(scores):.3f}")
```

`SplitComposer` chains `TargetedGroupSplit` (score only on the selected domain), `SemiSupervisedSplit` (keep unlabeled samples in every training fold), and `TransductiveSplit` (expose the evaluation inputs during fitting). The steps run inside an `imblearn.pipeline.Pipeline`, so you can pass metadata with `adapt__domains=...` to feed `MIDA`, while `SupervisedOnlyEstimator` drops any `-1` targets before invoking the wrapped estimator, letting you reuse scikit-learn components without altering their internals.

## Key features
- `TargetedGroupSplit` for focused evaluation on a single domain while keeping the rest of the data available for fitting.
- Compatibility layers that wrap scikit-learn models and pipelines so you can drop transductive components into existing projects.

## Getting help
- Open an issue on GitHub if something is unclear or broken.
- Contributions are welcome—feel free to submit pull requests with enhancements, documentation, or bug fixes.
