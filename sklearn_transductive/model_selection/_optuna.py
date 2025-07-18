from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from logging import DEBUG, INFO, WARNING
from numbers import Integral, Number
from time import time
from typing import Any, List, Union
import warnings

import numpy as np
from optuna import TrialPruned, distributions, logging, samplers
from optuna import study as study_module
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.distributions import _convert_old_distribution_to_new_distribution
from optuna.study import StudyDirection
from optuna.terminator import report_cross_validation_scores
from optuna.trial import FrozenTrial, Trial

# TODO (VALIDATE+TEST):: Override imports
with try_import() as _imports:
    import pandas as pd
    import scipy as sp
    from scipy.sparse import spmatrix
    import sklearn
    from sklearn.base import BaseEstimator, clone, is_classifier
    from sklearn.model_selection import BaseCrossValidator, check_cv
    from sklearn.utils import _safe_indexing as sklearn_safe_indexing
    from sklearn.utils import check_random_state
    from sklearn.utils._estimator_html_repr import _VisualBlock
    from sklearn.utils._metadata_requests import (
        MetadataRouter,
        MethodMapping,
        _routing_enabled,
    )
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.utils.validation import check_is_fitted

    from ..metrics import check_scoring
    from ._validation import cross_validate


if not _imports.is_successful():
    BaseEstimator = object  # NOQA

# Add export all
__all__ = ["OptunaSearchCV"]

ArrayLikeType = Union[List, np.ndarray, "pd.Series", "spmatrix"]
OneDimArrayLikeType = Union[List[float], np.ndarray, "pd.Series"]
TwoDimArrayLikeType = Union[List[List[float]], np.ndarray, "pd.DataFrame", "spmatrix"]
IterableType = Union[List, "pd.DataFrame", np.ndarray, "pd.Series", "spmatrix", None]
IndexableType = Union[Iterable, None]

_logger = logging.get_logger(__name__)


def _check_fit_params(X: TwoDimArrayLikeType, fit_params: dict, indices: OneDimArrayLikeType) -> dict:
    fit_params_validated = {}
    for key, value in fit_params.items():
        # NOTE Original implementation:
        # https://github.com/scikit-learn/scikit-learn/blob/ \
        # 2467e1b84aeb493a22533fa15ff92e0d7c05ed1c/sklearn/utils/validation.py#L1324-L1328
        # Scikit-learn does not accept non-iterable inputs.
        # This line is for keeping backward compatibility.
        # (See: https://github.com/scikit-learn/scikit-learn/issues/15805)
        if not _is_arraylike(value) or _num_samples(value) != _num_samples(X):
            fit_params_validated[key] = value
        else:
            fit_params_validated[key] = _make_indexable(value)
            fit_params_validated[key] = _safe_indexing(fit_params_validated[key], indices)
    return fit_params_validated


# NOTE Original implementation:
# https://github.com/scikit-learn/scikit-learn/blob/ \
# 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L131-L135
def _is_arraylike(x: Any) -> bool:
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


# NOTE Original implementation:
# https://github.com/scikit-learn/scikit-learn/blob/ \
# 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L217-L234
def _make_indexable(iterable: IterableType) -> IndexableType:
    tocsr_func = getattr(iterable, "tocsr", None)
    if tocsr_func is not None and sp.sparse.issparse(iterable):
        return tocsr_func(iterable)
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def _num_samples(x: ArrayLikeType) -> int:
    # NOTE For dask dataframes
    # https://github.com/scikit-learn/scikit-learn/blob/ \
    # 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L155-L158
    x_shape = getattr(x, "shape", None)
    if x_shape is not None:
        if isinstance(x_shape[0], Integral):
            return int(x_shape[0])

    try:
        return len(x)
    except TypeError:
        raise TypeError("Expected sequence or array-like, got %s." % type(x)) from None


def _safe_indexing(
    X: OneDimArrayLikeType | TwoDimArrayLikeType, indices: OneDimArrayLikeType
) -> OneDimArrayLikeType | TwoDimArrayLikeType:
    if X is None:
        return X

    return sklearn_safe_indexing(X, indices)


class _Objective:
    """Callable that implements objective function.

    Args:
        estimator:
            Object to use to fit the data. This is assumed to implement the
            scikit-learn estimator interface. Either this needs to provide
            ``score``, or ``scoring`` must be passed.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Distributions are assumed to implement the optuna distribution
            interface.

        X:
            Training data.

        y:
            Target variable.

        cv:
            Cross-validation strategy.

        enable_pruning:
            If :obj:`True`, pruning is performed in the case where the
            underlying estimator supports ``partial_fit``.

        error_score:
            Value to assign to the score if an error occurs in fitting. If
            ``"raise"``, the error is raised. If numeric,
            :class:`sklearn.exceptions.FitFailedWarning` is raised. This does not
            affect the refit step, which will always raise the error.

        fit_params:
            Parameters passed to ``fit`` one the estimator.

        groups:
            Group labels for the samples used while splitting the dataset into
            train/validation set.

        max_iter:
            Maximum number of epochs. This is only used if the underlying
            estimator supports ``partial_fit``.

        return_train_score:
            If :obj:`True`, training scores will be included. Computing
            training scores is used to get insights on how different
            hyperparameter settings impact the overfitting/underfitting
            trade-off. However computing training scores can be
            computationally expensive and is not strictly required to select
            the hyperparameters that yield the best generalization
            performance.

        scoring:
            Scorer function.
    """

    def __init__(
        self,
        estimator: "sklearn.base.BaseEstimator",
        param_distributions: Mapping[str, distributions.BaseDistribution],
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType | TwoDimArrayLikeType | None,
        cv: "BaseCrossValidator",
        enable_pruning: bool,
        error_score: Number | float | str,
        fit_params: dict[str, Any],
        groups: OneDimArrayLikeType | None,
        max_iter: int,
        return_train_score: bool,
        scoring: Callable[..., Number],
    ) -> None:
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.error_score = error_score
        self.estimator = estimator
        self.fit_params = fit_params
        self.groups = groups
        self.max_iter = max_iter
        self.param_distributions = param_distributions
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.X = X
        self.y = y

    def __call__(self, trial: Trial) -> float:
        estimator = clone(self.estimator)
        params = self._get_params(trial)

        estimator.set_params(**params)

        if self.enable_pruning:
            scores = self._cross_validate_with_pruning(trial, estimator)
        else:
            sklearn_version = sklearn.__version__.split(".")
            sklearn_major_version = int(sklearn_version[0])
            sklearn_minor_version = int(sklearn_version[1])
            try:
                # TODO (VALIDATE+TEST): Add handler for metadata routing
                if sklearn_major_version == 1 and sklearn_minor_version >= 4:
                    scores = cross_validate(
                        estimator,
                        self.X,
                        self.y,
                        cv=self.cv,
                        error_score=self.error_score,
                        params=self.fit_params,
                        groups=None if _routing_enabled() else self.groups,
                        return_train_score=self.return_train_score,
                        scoring=self.scoring,
                    )
                else:
                    scores = cross_validate(
                        estimator,
                        self.X,
                        self.y,
                        cv=self.cv,
                        error_score=self.error_score,
                        fit_params=self.fit_params,
                        groups=self.groups,
                        return_train_score=self.return_train_score,
                        scoring=self.scoring,
                    )
            except ValueError:
                n_splits = self.cv.get_n_splits(self.X, self.y, self.groups)
                fit_time = np.array([np.nan] * n_splits)
                score_time = np.array([np.nan] * n_splits)
                test_score = np.array([self.error_score if self.error_score is not None else np.nan] * n_splits)

                scores = {
                    "fit_time": fit_time,
                    "score_time": score_time,
                    "test_score": test_score,
                }

        self._store_scores(trial, scores)

        test_scores = scores["test_score"]
        scores_list = test_scores if isinstance(test_scores, list) else list(test_scores.tolist())
        try:
            report_cross_validation_scores(trial, scores_list)
        except ValueError as e:
            warn_msg = ("Failed to report cross validation scores for TerminatorCallback, with error: {}").format(e)
            warnings.warn(warn_msg)

        return trial.user_attrs["mean_test_score"]

    def _cross_validate_with_pruning(
        self, trial: Trial, estimator: "sklearn.base.BaseEstimator"
    ) -> Mapping[str, OneDimArrayLikeType]:
        if is_classifier(estimator):
            partial_fit_params = self.fit_params.copy()
            y = self.y.values if isinstance(self.y, pd.Series) else self.y
            if y is not None:
                classes = np.unique(y)
            else:
                classes = np.array([None])

            partial_fit_params.setdefault("classes", classes)

        else:
            partial_fit_params = self.fit_params

        n_splits = self.cv.get_n_splits(self.X, self.y, groups=self.groups)
        estimators = [clone(estimator) for _ in range(n_splits)]
        scores = {
            "fit_time": np.zeros(n_splits),
            "score_time": np.zeros(n_splits),
            "test_score": np.empty(n_splits),
        }

        if self.return_train_score:
            scores["train_score"] = np.empty(n_splits)

        for step in range(self.max_iter):
            for i, (train, test) in enumerate(self.cv.split(self.X, self.y, groups=self.groups)):
                out = list(
                    np.asarray(
                        self._partial_fit_and_score(estimators[i], train, test, partial_fit_params),
                        dtype=float,
                    ).tolist()
                )

                if self.return_train_score:
                    scores["train_score"][i] = out.pop(0)

                scores["test_score"][i] = out[0]
                scores["fit_time"][i] += out[1]
                scores["score_time"][i] += out[2]

            intermediate_value = np.nanmean(scores["test_score"])

            trial.report(intermediate_value, step=step)

            if trial.should_prune():
                self._store_scores(trial, scores)

                raise TrialPruned("trial was pruned at iteration {}.".format(step))

        return scores

    # TODO (VALIDATE+TEST): Add option to sample from list of parameters.
    def _get_params(self, trial: Trial) -> dict[str, Any]:
        param_distributions = self.param_distributions

        if isinstance(param_distributions, list):
            index = trial.suggest_int("param_group", 0, len(param_distributions) - 1)
            param_distributions = param_distributions[index]
            return {
                name: trial._suggest("__".join((f"group_{index:0=4d}", name)), distribution)
                for name, distribution in param_distributions.items()
            }

        return {name: trial._suggest(name, distribution) for name, distribution in param_distributions.items()}

    def _partial_fit_and_score(
        self,
        estimator: "sklearn.base.BaseEstimator",
        train: list[int],
        test: list[int],
        partial_fit_params: dict[str, Any],
    ) -> list[Number]:
        X_train, y_train = _safe_split(estimator, self.X, self.y, train)
        X_test, y_test = _safe_split(estimator, self.X, self.y, test, train_indices=train)

        start_time = time()

        try:
            estimator.partial_fit(X_train, y_train, **partial_fit_params)

        except Exception as e:
            if self.error_score == "raise":
                raise e

            elif isinstance(self.error_score, Number):
                fit_time = time() - start_time
                test_score = self.error_score
                score_time = 0.0

                if self.return_train_score:
                    train_score = self.error_score

            else:
                raise ValueError("error_score must be 'raise' or numeric.") from e

        else:
            fit_time = time() - start_time
            test_score = self.scoring(estimator, X_test, y_test)
            score_time = time() - fit_time - start_time

            if self.return_train_score:
                train_score = self.scoring(estimator, X_train, y_train)

        # Required for type checking but is never expected to fail.
        assert isinstance(fit_time, Number)
        assert isinstance(score_time, Number)

        ret = [test_score, fit_time, score_time]

        if self.return_train_score:
            ret.insert(0, train_score)

        return ret

    def _store_scores(self, trial: Trial, scores: Mapping[str, OneDimArrayLikeType]) -> None:
        for name, array in scores.items():
            if name in ["test_score", "train_score"]:
                for i, score in enumerate(array):
                    trial.set_user_attr("split{}_{}".format(i, name), score)

            trial.set_user_attr("mean_{}".format(name), np.nanmean(array))
            trial.set_user_attr("std_{}".format(name), np.nanstd(array))


@experimental_class("0.17.0")
class OptunaSearchCV(BaseEstimator):
    """Hyperparameter search with cross-validation.

    Args:
        estimator:
            Object to use to fit the data. This is assumed to implement the
            scikit-learn estimator interface. Either this needs to provide
            ``score``, or ``scoring`` must be passed.

        param_distributions:
            Dictionary where keys are parameters and values are distributions.
            Distributions are assumed to implement the optuna distribution
            interface.

        cv:
            Cross-validation strategy. Possible inputs for cv are:

            - :obj:`None`, to use the default 5-fold cross validation,
            - integer to specify the number of folds in a CV splitter,
            - `CV splitter <https://scikit-learn.org/stable/glossary.html#term-CV-splitter>`_,
            - an iterable yielding (train, validation) splits as arrays of indices.

            For integer, if ``estimator`` is a classifier and ``y`` is
            either binary or multiclass,
            :class:`sklearn.model_selection.StratifiedKFold` is used. otherwise,
            :class:`sklearn.model_selection.KFold` is used.

        enable_pruning:
            If :obj:`True`, pruning is performed in the case where the
            underlying estimator supports ``partial_fit``.

        error_score:
            Value to assign to the score if an error occurs in fitting. If
            ``"raise"``, the error is raised. If numeric,
            :class:`sklearn.exceptions.FitFailedWarning` is raised. This does not
            affect the refit step, which will always raise the error.

        max_iter:
            Maximum number of epochs. This is only used if the underlying
            estimator supports ``partial_fit``.

        n_jobs:
            Number of :obj:`threading` based parallel jobs. :obj:`None` means ``1``.
            ``-1`` means using the number is set to CPU count.

                .. note::
                    ``n_jobs`` allows parallelization using :obj:`threading` and may suffer from
                    `Python's GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_.
                    It is recommended to use `process-based optimization <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#distributed>`_
                    if ``func`` is CPU bound.

        n_trials:
            Number of trials. If :obj:`None`, there is no limitation on the
            number of trials. If ``timeout`` is also set to :obj:`None`,
            the study continues to create trials until it receives a
            termination signal such as Ctrl+C or SIGTERM. This trades off
            runtime vs quality of the solution.

        random_state:
            Seed of the pseudo random number generator. If int, this is the
            seed used by the random number generator. If
            :class:`numpy.random.RandomState` object, this is the random number
            generator. If :obj:`None`, the global random state from
            :mod:`numpy.random` is used.

        refit:
            If :obj:`True`, refit the estimator with the best found
            hyperparameters. The refitted estimator is made available at the
            ``best_estimator_`` attribute and permits using ``predict``
            directly.

        return_train_score:
            If :obj:`True`, training scores will be included. Computing
            training scores is used to get insights on how different
            hyperparameter settings impact the overfitting/underfitting
            trade-off. However computing training scores can be
            computationally expensive and is not strictly required to select
            the hyperparameters that yield the best generalization
            performance.

        scoring:
            String or callable to evaluate the predictions on the validation data.
            If :obj:`None`, ``score`` on the estimator is used.

        study:
            Study corresponds to the optimization task. If :obj:`None`, a new
            study is created.

        subsample:
            Proportion of samples that are used during hyperparameter search.

            - If int, then draw ``subsample`` samples.
            - If float, then draw ``subsample`` * ``X.shape[0]`` samples.

        timeout:
            Time limit in seconds for the search of appropriate models. If
            :obj:`None`, the study is executed without time limitation. If
            ``n_trials`` is also set to :obj:`None`, the study continues to
            create trials until it receives a termination signal such as
            Ctrl+C or SIGTERM. This trades off runtime vs quality of the
            solution.

        verbose:
            Verbosity level. The higher, the more messages.

        callbacks:
            List of callback functions that are invoked at the end of each trial. Each function
            must accept two parameters with the following types in this order:
            :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.

            .. seealso::

                See the tutorial of `Callback for Study.optimize <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html#optuna-callback>`_
                for how to use and implement callback functions.

        catch:
            A study continues to run even when a trial raises one of the exceptions specified
            in this argument. Default is an empty tuple, i.e. the study will stop for any
            exception except for :class:`~optuna.exceptions.TrialPruned`.

    Attributes:
        best_estimator_:
            Estimator that was chosen by the search. This is present only if
            ``refit`` is set to :obj:`True`.

        n_splits_:
            Number of cross-validation splits.

        refit_time_:
            Time for refitting the best estimator. This is present only if
            ``refit`` is set to :obj:`True`.

        sample_indices_:
            Indices of samples that are used during hyperparameter search.

        scorer_:
            Scorer function.

        study_:
            Actual study.

    Examples:

    .. note::
        By following the scikit-learn convention for scorers, the direction of optimization is
        ``maximize``. See https://scikit-learn.org/stable/modules/model_evaluation.html.
        For the minimization problem, please multiply ``-1``.
    """  # NOQA: E501

    _required_parameters = ["estimator", "param_distributions"]

    @property
    def _estimator_type(self) -> str:
        return self.estimator._estimator_type

    @property
    def best_index_(self) -> int:
        """Trial number which corresponds to the best candidate parameter setting.

        Returned value is equivalent to ``optuna_search.best_trial_.number``.
        """

        return self.best_trial_.number

    @property
    def best_params_(self) -> dict[str, Any]:
        """Parameters of the best trial in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self) -> float:
        """Mean cross-validated score of the best estimator."""

        self._check_is_fitted()

        return self.study_.best_value

    @property
    def best_trial_(self) -> FrozenTrial:
        """Best trial in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def classes_(self) -> OneDimArrayLikeType:
        """Class labels."""

        self._check_is_fitted()

        return self.best_estimator_.classes_

    @property
    def cv_results_(self) -> dict[str, Any]:
        """A dictionary mapping a metric name to a list of Cross-Validation results of all trials."""  # NOQA: E501

        cv_results_dict_in_list = [trial_.user_attrs for trial_ in self.trials_]
        if len(cv_results_dict_in_list) == 0:
            cv_results_list_in_dict = {}
        else:
            cv_results_list_in_dict = {
                key: [dict_[key] for dict_ in cv_results_dict_in_list] for key in cv_results_dict_in_list[0]
            }
        return cv_results_list_in_dict

    @property
    def n_trials_(self) -> int:
        """Actual number of trials."""

        return len(self.trials_)

    @property
    def trials_(self) -> list[FrozenTrial]:
        """All trials in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.trials

    @property
    def user_attrs_(self) -> dict[str, Any]:
        """User attributes in the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.user_attrs

    # TODO (VALIDATE+TEST): Add metadata routing
    # NOTE: This is only to allow the class to work with the metadata routing system.
    def get_metadata_routing(self):
        router = MetadataRouter(self.__class__.__name__)
        mapper = MethodMapping()
        mapper.add(caller="fit", callee="fit")
        mapper.add(caller="predict", callee="predict")
        mapper.add(caller="predict_proba", callee="predict_proba")
        mapper.add(caller="predict_log_proba", callee="predict_log_proba")
        mapper.add(caller="decision_function", callee="decision_function")
        mapper.add(caller="transform", callee="transform")
        mapper.add(caller="inverse_transform", callee="inverse_transform")
        mapper.add(caller="score", callee="score")
        router.add(estimator=self.estimator, method_mapping=mapper)
        return router

    def decision_function(self, X: TwoDimArrayLikeType, **kwargs: Any) -> OneDimArrayLikeType | TwoDimArrayLikeType:
        """Call ``decision_function`` on the best estimator.

        This is available only if the underlying estimator supports
        ``decision_function`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.decision_function(X, **kwargs)

    def inverse_transform(self, X: TwoDimArrayLikeType, *args: Any, **kwargs: Any) -> TwoDimArrayLikeType:
        """Call ``inverse_transform`` on the best estimator.

        This is available only if the underlying estimator supports
        ``inverse_transform`` and ``refit`` is set to :obj:`True`.
        Please check the following to know more about
        :meth:`sklearn.preprocessing.FunctionTransformer.inverse_transform`.
        """

        self._check_is_fitted()

        return self.best_estimator_.inverse_transform(X, *args, **kwargs)

    def predict(self, X: TwoDimArrayLikeType, **kwargs: Any) -> OneDimArrayLikeType | TwoDimArrayLikeType:
        """Call ``predict`` on the best estimator.

        This is available only if the underlying estimator supports ``predict``
        and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict(X, **kwargs)

    def predict_log_proba(self, X: TwoDimArrayLikeType, **kwargs: Any) -> TwoDimArrayLikeType:
        """Call ``predict_log_proba`` on the best estimator.

        This is available only if the underlying estimator supports
        ``predict_log_proba`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_log_proba(X, **kwargs)

    def predict_proba(self, X: TwoDimArrayLikeType, **kwargs: Any) -> TwoDimArrayLikeType:
        """Call ``predict_proba`` on the best estimator.

        This is available only if the underlying estimator supports
        ``predict_proba`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.predict_proba(X, **kwargs)

    @property
    def score_samples(self) -> Callable[..., OneDimArrayLikeType]:
        """Call ``score_samples`` on the best estimator.

        This is available only if the underlying estimator supports
        ``score_samples`` and ``refit`` is set to :obj:`True`.
        """

        self._check_is_fitted()

        return self.best_estimator_.score_samples

    @property
    def set_user_attr(self) -> Callable[..., None]:
        """Call ``set_user_attr`` on the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.set_user_attr

    def transform(self, X: TwoDimArrayLikeType, *args: Any, **kwargs: Any) -> TwoDimArrayLikeType:
        """Call ``transform`` on the best estimator.

        This is available only if the underlying estimator supports
        ``transform`` and ``refit`` is set to :obj:`True`.
        Please check the following to know more about
        :meth:`sklearn.preprocessing.FunctionTransformer.transform`
        """

        self._check_is_fitted()

        return self.best_estimator_.transform(X, *args, **kwargs)

    @property
    def trials_dataframe(self) -> Callable[..., "pd.DataFrame"]:
        """Call ``trials_dataframe`` on the :class:`~optuna.study.Study`."""

        self._check_is_fitted()

        return self.study_.trials_dataframe

    # TODO (VALIDATE+TEST): Handle checking for param_distributions to allow list of mappings
    def __init__(
        self,
        estimator: "sklearn.base.BaseEstimator",
        param_distributions: Mapping[str, distributions.BaseDistribution],
        *,
        cv: int | "BaseCrossValidator" | Iterable | None = None,
        enable_pruning: bool = False,
        error_score: Number | float | str = np.nan,
        max_iter: int = 1000,
        n_jobs: int | None = None,
        n_trials: int | None = 10,
        random_state: int | np.random.RandomState | None = None,
        refit: bool = True,
        return_train_score: bool = False,
        scoring: Callable[..., float] | str | None = None,
        study: study_module.Study | None = None,
        subsample: float | int = 1.0,
        timeout: float | None = None,
        verbose: int = 0,
        callbacks: list[Callable[[study_module.Study, FrozenTrial], None]] | None = None,
        catch: Iterable[type[Exception]] | type[Exception] = (),
    ) -> None:
        _imports.check()

        if not isinstance(param_distributions, (Mapping, list)):
            raise TypeError("param_distributions must be a mapping or a list of mapping.")

        if isinstance(param_distributions, list):
            for i, param_distribution in enumerate(param_distributions):
                if not isinstance(param_distribution, Mapping):
                    raise TypeError(f"param_distributions[{i}] must be a mapping.")
                for key, dist in param_distribution.items():
                    if dist != _convert_old_distribution_to_new_distribution(dist):
                        raise ValueError(
                            f"Deprecated distribution is specified in `{key}` of param_distributions. "
                            "Rejecting this because it may cause unexpected behavior. "
                            "Please use new distributions such as FloatDistribution etc."
                        )

        else:
            # Rejecting deprecated distributions as they may cause cryptic error
            # when cloning OptunaSearchCV instance.
            # https://github.com/optuna/optuna/issues/4084
            for key, dist in param_distributions.items():
                if dist != _convert_old_distribution_to_new_distribution(dist):
                    raise ValueError(
                        f"Deprecated distribution is specified in `{key}` of param_distributions. "
                        "Rejecting this because it may cause unexpected behavior. "
                        "Please use new distributions such as FloatDistribution etc."
                    )

        self.cv = cv
        self.enable_pruning = enable_pruning
        self.error_score = error_score
        self.estimator = estimator
        self.max_iter = max_iter
        self.n_trials = n_trials
        self.n_jobs = n_jobs if n_jobs else 1
        self.param_distributions = (
            param_distributions if isinstance(param_distributions, (dict, list)) else dict(param_distributions)
        )
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.scoring = scoring
        self.study = study
        self.subsample = subsample
        self.timeout = timeout
        self.verbose = verbose
        self.callbacks = callbacks
        self.catch = catch

    def _check_is_fitted(self) -> None:
        attributes = ["n_splits_", "sample_indices_", "scorer_", "study_"]

        if self.refit:
            attributes += ["best_estimator_", "refit_time_"]

        check_is_fitted(self, attributes)

    def _check_params(self) -> None:
        if not hasattr(self.estimator, "fit"):
            raise ValueError("estimator must be a scikit-learn estimator.")

        if isinstance(self.param_distributions, dict):

            for name, distribution in self.param_distributions.items():
                if not isinstance(distribution, distributions.BaseDistribution):
                    raise ValueError("Value of {} must be a optuna distribution.".format(name))
        else:
            for param_distributions in self.param_distributions:
                for name, distribution in param_distributions.items():
                    if not isinstance(distribution, distributions.BaseDistribution):
                        raise ValueError("Value of {} must be a optuna distribution.".format(name))

        if self.enable_pruning and not hasattr(self.estimator, "partial_fit"):
            raise ValueError("estimator must support partial_fit.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got {}.".format(self.max_iter))

        if self.study is not None and self.study.direction != StudyDirection.MAXIMIZE:
            raise ValueError("direction of study must be 'maximize'.")

    # TODO (VALIDATE+TEST): Add visual block for visualizing pipeline
    def _sk_visual_block_(self):
        if hasattr(self, "best_estimator_"):
            key, estimator = "best_estimator_", self.best_estimator_
        else:
            key, estimator = "estimator", self.estimator

        return _VisualBlock(
            "parallel",
            [estimator],
            names=[f"{key}: {estimator.__class__.__name__}"],
            name_details=[str(estimator)],
        )

    def _more_tags(self) -> dict[str, bool]:
        return {"non_deterministic": True, "no_validation": True}

    # TODO (VALIDATE+TEST): Add handler to refit from list of param group.
    def _refit(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType | TwoDimArrayLikeType | None = None,
        **fit_params: Any,
    ) -> "OptunaSearchCV":
        n_samples = _num_samples(X)

        self.best_estimator_ = clone(self.estimator)

        try:
            best_params = self.study_.best_params
            if "param_group" in best_params:
                best_params.pop("param_group")
                best_params = {"__".join(k.split("__")[1:]): v for k, v in best_params.items()}

            self.best_estimator_.set_params(**best_params)
        except ValueError as e:
            _logger.exception(e)

        _logger.info("Refitting the estimator using {} samples...".format(n_samples))

        start_time = time()

        self.best_estimator_.fit(X, y, **fit_params)

        self.refit_time_ = time() - start_time

        _logger.info("Finished refitting! (elapsed time: {:.3f} sec.)".format(self.refit_time_))

        return self

    def fit(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType | TwoDimArrayLikeType | None = None,
        groups: OneDimArrayLikeType | None = None,
        **fit_params: Any,
    ) -> "OptunaSearchCV":
        """Run fit with all sets of parameters.

        Args:
            X:
                Training data.

            y:
                Target variable.

            groups:
                Group labels for the samples used while splitting the dataset
                into train/validation set.

            **fit_params:
                Parameters passed to ``fit`` on the estimator.

        Returns:
            self.
        """

        self._check_params()

        random_state = check_random_state(self.random_state)
        max_samples = self.subsample
        n_samples = _num_samples(X)
        old_level = _logger.getEffectiveLevel()

        if self.verbose > 1:
            _logger.setLevel(DEBUG)
        elif self.verbose > 0:
            _logger.setLevel(INFO)
        else:
            _logger.setLevel(WARNING)

        self.sample_indices_ = np.arange(n_samples)

        if isinstance(max_samples, float):
            max_samples = int(max_samples * n_samples)

        if max_samples < n_samples:
            self.sample_indices_ = random_state.choice(self.sample_indices_, max_samples, replace=False)

            self.sample_indices_.sort()

        X_res = _safe_indexing(X, self.sample_indices_)
        y_res = _safe_indexing(y, self.sample_indices_)
        groups_res = _safe_indexing(groups, self.sample_indices_)
        fit_params_res = fit_params

        if fit_params_res is not None:
            fit_params_res = _check_fit_params(X, fit_params, self.sample_indices_)

        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y_res, classifier=classifier)

        self.n_splits_ = cv.get_n_splits(X_res, y_res, groups=groups_res)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if self.study is None:
            seed = random_state.randint(0, np.iinfo("int32").max)
            sampler = samplers.TPESampler(seed=seed)

            self.study_ = study_module.create_study(direction="maximize", sampler=sampler)

        else:
            self.study_ = self.study

        objective = _Objective(
            self.estimator,
            self.param_distributions,
            X_res,
            y_res,
            cv,
            self.enable_pruning,
            self.error_score,
            fit_params_res,
            groups_res,
            self.max_iter,
            self.return_train_score,
            self.scorer_,
        )

        _logger.info(
            "Searching the best hyperparameters using {} " "samples...".format(_num_samples(self.sample_indices_))
        )

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=self.callbacks,
            catch=self.catch,
        )

        _logger.info("Finished hyperparameter search!")

        if self.refit:
            self._refit(X, y, **fit_params)

        _logger.setLevel(old_level)

        return self

    def score(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType | TwoDimArrayLikeType | None = None,
        predict_params=None,
    ) -> float:
        """Return the score on the given data.

        Args:
            X:
                Data.

            y:
                Target variable.

        Returns:
            Scaler score.
        """

        return self.scorer_(self.best_estimator_, X, y, predict_params=predict_params)
