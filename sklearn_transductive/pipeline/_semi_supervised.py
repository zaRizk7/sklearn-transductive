import numpy as np
from sklearn.base import BaseEstimator, _fit_context
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import _safe_indexing, get_tags, safe_mask
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

from ..utils.metaestimators import subestimator_has

__all__ = ["SemiSupervisedEstimator"]


# Just used for ignoring unlabeled data with -1 labels
# One example is to compose multiple estimators that may have
# unlabeled data in the middle of the pipeline with a transformation
# that may have transductive properties.
class SemiSupervisedEstimator(BaseEstimator):
    _parameter_constraints: dict = {"estimator": [HasMethods(["fit"])]}

    def __init__(self, estimator=FunctionTransformer()):
        self.estimator = estimator

    @property
    def classes_(self):
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
        check_is_fitted(self)
        routed_params = self._process_routing(params, "transform")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.transform(X, **routed_params)

    @available_if(subestimator_has("estimator", "inverse_transform"))
    def inverse_transform(self, X, **params):
        check_is_fitted(self)
        routed_params = self._process_routing(params, "inverse_transform")
        return self.estimator.inverse_transform(X, **routed_params)

    def fit_transform(self, X, y=None, **params):
        if y is None:
            self.fit(X, **params)
        else:
            self.fit(X, y, **params)
        return self.transform(X, **params)

    @available_if(subestimator_has("estimator", "predict"))
    def predict(self, X, **params):
        check_is_fitted(self)
        routed_params = self._process_routing(params, "predict")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.predict(X, **routed_params)

    @available_if(subestimator_has("estimator", "predict_proba"))
    def predict_proba(self, X, **params):
        check_is_fitted(self)
        routed_params = self._process_routing(params, "predict_proba")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.predict_proba(X, **routed_params)

    @available_if(subestimator_has("estimator", "predict_log_proba"))
    def predict_log_proba(self, X, **params):
        check_is_fitted(self)
        routed_params = self._process_routing(params, "predict_log_proba")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.predict_log_proba(X, **routed_params)

    @available_if(subestimator_has("estimator", "decision_function"))
    def decision_function(self, X, **params):
        check_is_fitted(self)
        routed_params = self._process_routing(params, "decision_function")
        X = validate_data(self, X, accept_sparse=True, reset=False)
        return self.estimator.decision_function(X, **routed_params)
