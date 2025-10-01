import numpy as np
from sklearn.model_selection import check_cv
from sklearn.utils import safe_mask
from sklearn.utils._indexing import _safe_indexing
from sklearn.utils._metadata_requests import _MetadataRequester
from sklearn.utils.validation import _num_samples, indexable

__all__ = [
    "SplitComposer",
    "TransductiveSplit",
    "AggregateGroupLabel",
    "SemiSupervisedSplit",
    "TargetedGroupSplit",
]


class TransductiveSplit(_MetadataRequester):
    """Ensure every held-out sample is also available for fitting.

    The wrapped cross-validator determines the testing folds. This adapter then
    unions the base train and test indices so that the test samples are also
    present in the training set. The behaviour departs from scikit-learn's
    standard splitters, which normally keep test folds strictly disjoint from
    the training folds, and is useful in transductive learning where the model
    may access the evaluation inputs during fitting.

    Parameters
    ----------
    cv : int, cross-validator instance or ``None``
        Base splitter supplied to :func:`~sklearn.model_selection.check_cv`. If
        ``None``, a default stratified splitter is chosen for classification
        problems.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> from sklearn_transductive.model_selection import TransductiveSplit
    >>> splitter = TransductiveSplit(cv=KFold(n_splits=3, shuffle=True, random_state=0))
    >>> X = np.zeros((6, 2))
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>> for train, test in splitter.split(X, y):
    ...     assert set(test).issubset(train)
    """

    __metadata_request__split = {"groups": True}

    def __init__(self, cv=None):
        self.cv = cv

    def split(self, X, y=None, groups=None):
        cv = check_cv(self.cv, y, classifier=True)
        for train, test in cv.split(X, y, groups):
            train = np.union1d(train, test)
            yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        cv = check_cv(self.cv, y, classifier=True)
        return cv.get_n_splits(X, y, groups)


class AggregateGroupLabel(_MetadataRequester):
    """Collapse labels and groups before delegating to base CV.

    scikit-learn splitters treat ``groups`` independently from ``y``. This
    wrapper builds composite labels combining the target and group information,
    while optionally collapsing the ``ignore_y`` class into a single bucket. The
    resulting aggregated labels are then passed to the underlying splitter so
    that its stratification or grouping rules operate on the merged view.

    Parameters
    ----------
    cv : int, cross-validator instance or ``None``
        Base splitter provided to :func:`~sklearn.model_selection.check_cv`.
    ignore_y : int or hashable, default=-1
        Label value that is kept in a dedicated bucket across groups.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sklearn_transductive.model_selection import AggregateGroupLabel
    >>> y = np.array([0, 0, 1, 1, -1, -1])
    >>> groups = np.array(["a", "b", "a", "b", "a", "b"])
    >>> splitter = AggregateGroupLabel(cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0))
    >>> for train, test in splitter.split(np.zeros_like(y), y, groups):
    ...     # Each fold stratifies on (label, group) except for ignore_y
    ...     assert set(test) <= set(range(len(y)))
    """

    __metadata_request__split = {"groups": True}

    def __init__(self, cv=None, ignore_y=-1):
        self.cv = cv
        self.ignore_y = ignore_y

    def _get_group_labels(self, X, y=None, groups=None):
        if y is None:
            y = np.zeros(X.shape[0])

        if groups is None:
            groups = np.zeros(X.shape[0])

        agg = np.column_stack([y, groups])
        _, agg = np.unique(agg, axis=0, return_inverse=True)

        mask = safe_mask(agg, y == self.ignore_y)
        agg[mask] = self.ignore_y
        return agg

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y, classifier=True)
        agg = self._get_group_labels(X, y, groups)
        for train, test in cv.split(X, agg, agg):
            yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y, classifier=True)
        agg = self._get_group_labels(X, y, groups)
        return cv.get_n_splits(X, agg, agg)


class SemiSupervisedSplit(_MetadataRequester):
    """Run CV on labeled samples while keeping unlabeled data in train folds.

    scikit-learn splitters assume every sample has a label. This adapter
    isolates the labeled subset (``y != -1``), runs the wrapped cross-validator
    on that subset, and always merges the unlabeled indices back into the
    training fold. The test splits therefore evaluate only on labeled data while
    the estimator still sees the unlabeled examples during fitting.

    Parameters
    ----------
    cv : int, cross-validator instance or ``None``
        Base splitter supplied to :func:`~sklearn.model_selection.check_cv` for
        the labeled subset.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sklearn_transductive.model_selection import SemiSupervisedSplit
    >>> y = np.array([0, 0, 1, 1, -1, -1])
    >>> splitter = SemiSupervisedSplit(cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0))
    >>> for train, test in splitter.split(np.zeros((len(y), 1)), y):
    ...     assert set(test).issubset({0, 1, 2, 3})  # only labeled samples
    ...     assert {4, 5}.issubset(train)  # unlabeled always in train
    """

    __metadata_request__split = {"groups": True}

    def __init__(self, cv=None):
        self.cv = cv

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y, classifier=True)

        num_samples = _num_samples(y)
        unlabeled = y == -1
        indices = np.arange(num_samples)

        unlabeled_indices = _safe_indexing(indices, unlabeled)
        labeled_indices = _safe_indexing(indices, ~unlabeled)
        y_labeled = _safe_indexing(y, ~unlabeled)
        groups_labeled = _safe_indexing(groups, ~unlabeled)

        for train, test in cv.split(y_labeled, y_labeled, groups_labeled):
            train = _safe_indexing(labeled_indices, train)
            train = np.union1d(unlabeled_indices, train)
            test = _safe_indexing(labeled_indices, test)

            yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        cv = check_cv(self.cv, y, classifier=True)
        return cv.get_n_splits(X, y, groups)


class SplitComposer(_MetadataRequester):
    """Sequentially wrap splitters to modify behaviour step by step.

    Each element in ``steps`` receives the previously composed splitter as its
    ``cv`` argument. This allows chaining adapters such as
    :class:`TransductiveSplit` or :class:`AggregateGroupLabel` around a base
    scikit-learn splitter without rewriting their logic. The resulting composed
    splitter presents the same interface as the original cross-validator but
    applies the stacked overrides when generating splits.

    Parameters
    ----------
    cv : int, cross-validator instance or ``None``
        Initial base splitter that will be wrapped by the listed steps.
    steps : list of _MetadataRequester instances
        Adapters applied in order; the last element runs the actual ``split``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sklearn_transductive.model_selection import (
    ...     AggregateGroupLabel,
    ...     SemiSupervisedSplit,
    ...     SplitComposer,
    ... )
    >>> base = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    >>> composer = SplitComposer(
    ...     cv=base,
    ...     steps=[SemiSupervisedSplit(), AggregateGroupLabel(ignore_y=-1)],
    ... )
    >>> X = np.zeros((6, 1))
    >>> y = np.array([0, 0, 1, 1, -1, -1])
    >>> groups = np.array(["a", "b", "a", "b", "a", "b"])
    >>> for train, test in composer.split(X, y, groups):
    ...     assert set(test).issubset({0, 1, 2, 3})
    """

    def __init__(self, cv=None, steps=None):
        self.cv = cv
        self.steps = steps

    def _compose(self, y):
        cv = check_cv(self.cv, y, classifier=True)
        for step in reversed(self.steps):
            step.cv = cv
            cv = step
        return cv

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        cv = self._compose(y)

        for train, test in cv.split(X, y, groups):
            yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        cv = self._compose(y)
        return cv.get_n_splits(X, y, groups)


# Take PredefinedSplit as a reference
# We want to have a split wrapper that can select a portion of the data
# from the group labels and do CV on the selected data
# Note we prefer this to be a wrapper rather than a new class
# The split must enforce excluded data to be in the training set
# The split must be able to handle the case where the group labels are not available
# Path: sklearn/model_selection/_split.py
class TargetedGroupSplit(_MetadataRequester):
    """Wrapper that focuses cross-validation on specific groups.

    This splitter wraps any scikit-learn compatible cross-validator and limits
    the test folds to the samples whose ``groups`` value matches
    ``target_group``. Samples outside of the targeted groups are always added to
    the training indices, enabling transductive or domain adaptation scenarios
    where a source domain should remain available for model fitting while the
    evaluation focuses on a particular target domain.

    Parameters
    ----------
    cv : int, cross-validator instance or ``None``
        The base cross-validator that will run on the targeted samples. If
        ``None``, :func:`~sklearn.model_selection.check_cv` chooses a sensible
        default for classification tasks (typically ``StratifiedKFold``).
    target_group : array-like of shape (n_groups,), hashable or ``None``
        Groups that should be used to build the test folds. If ``None`` or
        ``groups`` is ``None`` at split time, the base cross-validator operates
        on the full dataset.

    Notes
    -----
    ``TargetedGroupSplit`` inherits the metadata request interface so that the
    outer pipelines know that ``groups`` are required. The splitter mirrors the
    behaviour of :class:`~sklearn.model_selection.PredefinedSplit` in the sense
    that it wraps an existing cross-validator instead of re-implementing the
    splitting logic from scratch.

    Examples
    --------
    Run 3-fold CV only on the ``"target"`` domain while keeping the ``"source"``
    domain in every training fold::

        from sklearn.model_selection import KFold
        from sklearn_transductive.model_selection import TargetedGroupSplit

        groups = np.array(["source", "source", "target", "target"])
        splitter = TargetedGroupSplit(cv=KFold(n_splits=3), target_group="target")
        for train, test in splitter.split(X, y, groups):
            assert set(groups[test]) == {"target"}

    """

    __metadata_request__split = {"groups": True}

    def __init__(self, cv=None, target_group=None):
        self.cv = cv
        self.target_group = target_group

    def split(self, X, y=None, groups=None):
        """Generate indices while overriding the base CV behaviour.

        The wrapped cross-validator normally draws both train and test folds
        from the same pool of indices. This override restricts the test folds to
        the samples that belong to ``target_group`` and always re-injects the
        remaining samples into the training indices. If no ``groups`` or
        ``target_group`` are provided, the underlying scikit-learn splitter is
        used unchanged.
        """
        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y, classifier=True)
        idx = np.arange(_num_samples(y))
        # Generate mask for target group used for CV

        # Assume that if we do not define any group
        # we will do normal CV with the complete data
        target_group = self.target_group
        if target_group is None or groups is None:
            target_group = np.unique(groups)

        # Filter target group
        m_tgt = np.isin(groups, target_group)
        m_src = ~m_tgt
        # Index targeted group
        idx_tgt = _safe_indexing(idx, m_tgt)
        y_tgt = _safe_indexing(y, m_tgt)
        groups_tgt = _safe_indexing(groups, m_tgt)
        # Index source group
        idx_src = _safe_indexing(idx, m_src)
        for train, test in cv.split(idx_tgt, y_tgt, groups_tgt):
            # Add source group to training set
            train = _safe_indexing(idx_tgt, train)
            train = np.union1d(train, idx_src)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of base CV splits."""
        return self.cv.get_n_splits(X, y, groups)
