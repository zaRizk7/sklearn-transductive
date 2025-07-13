import numpy as np
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
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
    __metadata_request__split = {"groups": True}

    def __init__(self, cv=None, target_group=None):
        self.cv = cv
        self.target_group = target_group

    def split(self, X, y=None, groups=None):
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
        return self.cv.get_n_splits(X, y, groups)
