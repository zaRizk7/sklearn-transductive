import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils._set_output import _SetOutputMixin
from sklearn.utils.metadata_routing import _routing_enabled

from ..base import CovariateValidationMixin, DomainValidationMixin

__all__ = ["AdapterMixin", "BaseAdapter"]


class AdapterMixin(DomainValidationMixin, CovariateValidationMixin, _SetOutputMixin):
    def fit_transform(self, X, y=None, domains=None, covariates=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        domains : array-like of shape (n_samples,), default=None
            Group labels where the samples are sourced from.

        covariates : array-like of shape (n_samples, n_covariates), default=None
            Additional variables whose effects should be removed from `X`.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm

        # we do not route parameters here, since consumers don't route. But
        # since it's possible for a `transform` method to also consume
        # metadata, we check if that's the case, and we raise a warning telling
        # users that they should implement a custom `fit_transform` method
        # to forward metadata to `transform` as well.
        #
        # For that, we calculate routing and check if anything would be routed
        # to `transform` if we were to route them.
        if _routing_enabled():
            transform_params = self.get_metadata_routing().consumes(method="transform", params=fit_params.keys())
            if transform_params:
                warnings.warn(
                    (
                        f"This object ({self.__class__.__name__}) has a `transform`"
                        " method which consumes metadata, but `fit_transform` does not"
                        " forward metadata to `transform`. Please implement a custom"
                        " `fit_transform` method to forward metadata to `transform` as"
                        " well. Alternatively, you can explicitly do"
                        " `set_transform_request`and set all values to `False` to"
                        " disable metadata routed to `transform`, if that's an option."
                    ),
                    UserWarning,
                )

        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            self.fit(X, domains=domains, covariates=covariates, **fit_params)
        else:
            # fit method of arity 2 (supervised transformation)
            self.fit(X, y, domains, covariates, **fit_params)

        return self.transform(X, domains, covariates)


class BaseAdapter(AdapterMixin, BaseEstimator):
    # TODO: Add default metadata requests
    __metadata_request__fit = {"domains": True, "covariates": True}
    __metadata_request__transform = {"domains": True, "covariates": True}
    __metadata_request__fit_transform = {"domains": True, "covariates": True}
    __metadata_request__inverse_transform = {"domains": True, "covariates": True}

    def _augment(self, X=None, domains=None, covariates=None):
        fn = lambda item: item is not None  # noqa: E731
        A = tuple(filter(fn, (X, domains, covariates)))

        if len(A) > 0:
            return np.column_stack(A)

        msg = "At least one of X, domains, or covariates must be provided."
        raise ValueError(msg)
