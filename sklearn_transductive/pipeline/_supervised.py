import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import _safe_indexing, get_tags
from sklearn.utils._metadata_requests import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    _check_method_params,
    check_is_fitted,
    validate_data,
)

from ..utils.metaestimators import subestimator_has  # type: ignore[misc]

__all__ = ["SupervisedOnlyEstimator"]


# Just used for ignoring unlabeled data with -1 labels
# One example is to compose multiple estimators that may have
# unlabeled data in the middle of the pipeline with a transformation
# that may have transductive properties.
class SupervisedOnlyEstimator(BaseEstimator):
    """Wrap an estimator to ignore unlabeled samples when fitting.

    Samples whose target equals ``-1`` are skipped during :meth:`fit` while the
    other methods are transparently delegated to the wrapped estimator. This is
    useful in composite models that may temporarily introduce unlabeled
    observations but still rely on purely supervised components downstream.

    Parameters
    ----------
    estimator : estimator object, default=FunctionTransformer()
        Estimator implementing at least a ``fit`` method. Any additional
        methods (for example ``transform`` or ``predict``) will be delegated to
        this estimator when available.
    """

    _parameter_constraints: dict = {"estimator": [HasMethods(["fit"])]}

    def __init__(self, estimator=FunctionTransformer()):
        self.estimator = estimator

    @property
    def classes_(self):
        """Class labels encountered during :meth:`fit`.

        Returns
        -------
        ndarray of shape (n_classes,)
            Class labels exposed by the wrapped estimator. The attribute is
            available whenever the underlying estimator defines ``classes_``
            after fitting.
        """
        check_is_fitted(self)
        return self.estimator.classes_

    def set_params(self, **params):
        params = {f"estimator__{k}": v for k, v in params.items() if k != "estimator"}
        return super().set_params(**params)

    def get_metadata_routing(self):
        router = MetadataRouter(self.__class__.__name__)
        mapper = MethodMapping()
        if hasattr(self.estimator, "fit_transform"):
            mapper.add(caller="fit", callee="fit_transform")
            mapper.add(caller="fit_transform", callee="fit_transform")
        else:
            mapper.add(caller="fit", callee="fit")
            mapper.add(caller="fit", callee="transform")
            mapper.add(caller="fit_transform", callee="fit")
            mapper.add(caller="fit_transform", callee="transform")
        mapper.add(caller="transform", callee="transform")
        mapper.add(caller="inverse_transform", callee="inverse_transform")
        mapper.add(caller="predict", callee="predict")
        mapper.add(caller="predict_proba", callee="predict_proba")
        mapper.add(caller="predict_log_proba", callee="predict_log_proba")
        mapper.add(caller="decision_function", callee="decision_function")
        mapper.add(caller="score", callee="score")
        if hasattr(self.estimator, "score_samples"):
            mapper.add(caller="score_samples", callee="score_samples")
        router.add(estimator=self.estimator, method_mapping=mapper)
        return router

    def __sklearn_tags__(self):
        return get_tags(self.estimator)

    def _process_routing(self, params, method):
        if _routing_enabled():
            routed_params = process_routing(self, method, **params)
            routed_params = routed_params.estimator.get(method, {})
        else:
            routed_params = params
        return routed_params

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **params):
        """Fit the wrapped estimator using labeled samples only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,), default=None
            Target values. Samples with ``y == -1`` are treated as unlabeled and
            ignored when fitting the wrapped estimator.
        **params : dict
            Additional parameters to pass to :meth:`estimator.fit`.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        routed_params = self._process_routing(params, "fit")
        X, y = validate_data(self, X, y, accept_sparse=True)

        (labeled,) = np.nonzero(y != -1)
        routed_params = _check_method_params(X, routed_params, labeled)
        X = _safe_indexing(X, labeled)
        y = _safe_indexing(y, labeled)

        self.estimator.fit(X, y, **routed_params)
        return self

    @available_if(subestimator_has("estimator", "transform"))
    def transform(self, X, **params):
        """Transform X using the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        **params : dict
            Additional parameters to pass to :meth:`estimator.transform`.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_transformed)
            Transformed samples as returned by the wrapped estimator.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "transform")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.transform(X, **routed_params)

    @available_if(subestimator_has("estimator", "inverse_transform"))
    def inverse_transform(self, X, **params):
        """Apply the inverse transformation of the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_transformed)
            Input samples in the transformed space.
        **params : dict
            Additional parameters to pass to
            :meth:`estimator.inverse_transform`.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Inverse transformed samples.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "inverse_transform")
        return self.estimator.inverse_transform(X, **routed_params)

    def fit_transform(self, X, y=None, **params):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,), default=None
            Target values. Samples with ``y == -1`` are ignored during fitting.
        **params : dict
            Additional parameters routed to both :meth:`fit` and
            :meth:`transform` when supported.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_transformed)
            Transformed samples as produced by the wrapped estimator.
        """
        if y is None:
            self.fit(X, **params)
        else:
            self.fit(X, y, **params)
        return self.transform(X, **params)

    @available_if(subestimator_has("estimator", "predict"))
    def predict(self, X, **params):
        """Predict target values for X using the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        **params : dict
            Additional parameters to pass to :meth:`estimator.predict`.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "predict")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.predict(X, **routed_params)

    @available_if(subestimator_has("estimator", "predict_proba"))
    def predict_proba(self, X, **params):
        """Estimate class probabilities for X using the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        **params : dict
            Additional parameters to pass to :meth:`estimator.predict_proba`.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "predict_proba")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.predict_proba(X, **routed_params)

    @available_if(subestimator_has("estimator", "predict_log_proba"))
    def predict_log_proba(self, X, **params):
        """Estimate log-probabilities for X using the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        **params : dict
            Additional parameters to pass to
            :meth:`estimator.predict_log_proba`.

        Returns
        -------
        y_log_prob : ndarray of shape (n_samples, n_classes)
            Predicted log-probabilities of each class.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "predict_log_proba")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.predict_log_proba(X, **routed_params)

    @available_if(subestimator_has("estimator", "decision_function"))
    def decision_function(self, X, **params):
        """Compute the decision function of X using the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        **params : dict
            Additional parameters to pass to
            :meth:`estimator.decision_function`.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values as returned by the wrapped estimator.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "decision_function")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.decision_function(X, **routed_params)

    @available_if(subestimator_has("estimator", "score"))
    def score(self, X, y=None, **params):
        """Score the wrapped estimator on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,), default=None
            True labels for ``X``.
        **params : dict
            Additional parameters to pass to :meth:`estimator.score`.

        Returns
        -------
        score : float
            Score returned by the wrapped estimator.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "score")
        X, y = validate_data(self, X, y, accept_sparse=True, reset=False)
        return self.estimator.score(X, y, **routed_params)

    @available_if(subestimator_has("estimator", "score_samples"))
    def score_samples(self, X, **params):
        """Compute the score samples of X using the wrapped estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        **params : dict
            Additional parameters to pass to :meth:`estimator.score_samples`.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Point-wise scores as returned by the wrapped estimator.
        """
        check_is_fitted(self)
        routed_params = self._process_routing(params, "score_samples")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.score_samples(X, **routed_params)
