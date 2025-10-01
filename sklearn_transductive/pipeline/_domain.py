import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import _safe_indexing
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data

from ..base import DomainValidationMixin
from ..utils.metaestimators import subestimator_has

__all__ = ["DomainAwareTransformer"]


class DomainAwareTransformer(TransformerMixin, DomainValidationMixin, BaseEstimator):
    """Fit independent transformers for each domain label when available.

    When domain metadata is passed at fit time, the wrapped ``transformer`` is
    cloned and trained separately for every observed domain. Subsequent calls
    to :meth:`transform` and :meth:`inverse_transform` dispatch the
    corresponding domain-specific estimator. If no domain information is
    provided, the transformer behaves exactly like the wrapped estimator.

    Parameters
    ----------
    transformer : estimator object, default=FunctionTransformer()
        Base transformer implementing ``fit`` and ``transform``. Any optional
        ``inverse_transform`` implementation is forwarded when available.
    """

    __metadata_request__fit = {"domains": True}
    __metadata_request__transform = {"domains": True}
    __metadata_request__fit_transform = {"domains": True}
    __metadata_request__inverse_transform = {"domains": True}

    _parameter_constraints: dict = {"transformer": [HasMethods(["fit", "transform", "fit_transform"])]}

    def __init__(self, transformer=FunctionTransformer()):
        self.transformer = transformer

    def set_params(self, **params):
        """Set parameters on the wrapped transformer using a flat namespace."""
        params = {f"transformer__{k}": v for k, v in params.items()}
        return super().set_params(**params)

    def _iter_domains(self, domains):
        """Yield index-array/domain pairs in the order they appear."""
        for domain in np.unique(domains):
            (indices,) = np.nonzero(domains == domain)
            if len(indices) <= 0:
                continue
            yield indices, domain

    def _sk_visual_block_(self):
        """Return the HTML representation block for sklearn visualizations."""
        estimators = []
        names = []
        name_details = []

        if self.fit_has_domain_:
            for domain, transformer in self.transformer_.items():
                estimators.append(transformer)
                names.append(f"transformer[{domain}]: {transformer.__class__.__name__}")
                name_details.append(str(transformer))
        elif hasattr(self, "transformer_"):
            estimators.append(self.transformer_)
            names.append(f"transformer: {self.transformer_.__class__.__name__}")
            name_details.append(str(self.transformer_))
        else:
            estimators.append(self.transformer)
            names.append(f"transformer: {self.transformer.__class__.__name__}")
            name_details.append(str(self.transformer))

        return _VisualBlock("parallel", estimators, names=names, name_details=name_details)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, domains=None):
        """Fit the transformer, optionally per domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,), default=None
            Optional target values routed to the underlying transformer.
        domains : array-like of shape (n_samples,), default=None
            Domain labels associated with each sample. When provided, an
            independent clone of ``transformer`` is trained for each unique
            value.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X, y = validate_data(self, X, y, accept_sparse=True)

        domains = self._validate_domains(domains, X, required=False, return_value="validated")

        self.fit_has_domain_ = domains is not None

        if self.fit_has_domain_:
            self.transformer_ = {}
            for indices, domain in self._iter_domains(domains):
                X_domain = _safe_indexing(X, indices)
                y_domain = _safe_indexing(y, indices)
                transformer = clone(self.transformer)
                transformer.fit(X_domain, y_domain)
                self.transformer_[domain] = transformer
        else:
            self.transformer_ = clone(self.transformer)
            self.transformer_.fit(X, y)

        return self

    def transform(self, X, domains=None):
        """Transform input samples using domain-specific estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to transform.
        domains : array-like of shape (n_samples,), default=None
            Domain labels for each sample. Required if domain information was
            supplied during :meth:`fit`.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_transformed)
            Transformed samples reordered to match the input order.
        """
        return self._method_call("transform", X, domains)

    @available_if(subestimator_has("transformer", "inverse_transform"))
    def inverse_transform(self, X, domains=None):
        """Apply the inverse transformation respecting domain boundaries.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_transformed)
            Samples expressed in the transformed space.
        domains : array-like of shape (n_samples,), default=None
            Domain labels for each sample. Required if domain information was
            supplied during :meth:`fit`.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Inverse transformed samples in the original feature space.
        """
        return self._method_call("inverse_transform", X, domains)

    def _method_call(self, method_name, X, domains=None):
        """Dispatch ``method_name`` to the appropriate domain-aware estimator."""
        check_is_fitted(self)
        if method_name == "transform" and self.fit_has_domain_:
            X = validate_data(self, X, accept_sparse=True, reset=False)

        domains = self._validate_domains(
            domains,
            X,
            required=self.fit_has_domain_,
            return_value="validated",
            reset=False,
        )

        if not self.fit_has_domain_:
            return getattr(self.transformer_, method_name)(X)

        X_ = []
        indices_ = []
        for indices, domain in self._iter_domains(domains):
            if domain not in self.transformer_:
                raise ValueError(f"Domain {domain} is not defined during 'fit'.")

            X_domain = _safe_indexing(X, indices)
            transformer = self.transformer_[domain]
            method = getattr(transformer, method_name)

            X_.append(method(X_domain))
            indices_.append(indices)

        X_ = np.row_stack(X_)
        indices_ = np.concatenate(indices_)
        indices = np.argsort(indices_)
        return _safe_indexing(X_, indices)

    def fit_transform(self, X, y=None, domains=None):
        """Fit to data, then transform it with domain awareness.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,), default=None
            Optional target values routed to the underlying transformer.
        domains : array-like of shape (n_samples,), default=None
            Domain labels to use when fitting and transforming.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_transformed)
            Transformed samples reordered to match the input order.
        """
        if y is None:
            self.fit(X, domains=domains)
        else:
            self.fit(X, y, domains=domains)
        return self.transform(X, domains=domains)
