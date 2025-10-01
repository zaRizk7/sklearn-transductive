"""
Based on the implementation from https://github.com/Warvito/neurocombat_sklearn
Credit goes to the original code author: Walter Hugo Lopez Pinaya

This is a modified implementation that is more compatible with the later version of
scikit-learn API. Additionally, this implementation is more flexible and have more
vectorized operations.
"""

from numbers import Integral, Real

import numpy as np
from scipy import linalg as la
from sklearn.base import OneToOneFeatureMixin, _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseAdapter

__all__ = ["ComBat"]


def _estimate_priors(gamma_hat, delta_hat):
    """Estimate empirical Bayes priors for the ComBat model.

    Parameters
    ----------
    gamma_hat : ndarray of shape (n_domains, n_features)
        Least-squares estimates of the domain-specific location parameters.
    delta_hat : ndarray of shape (n_domains, n_features)
        Least-squares estimates of the domain-specific scale parameters.

    Returns
    -------
    gamma_bar : ndarray of shape (n_domains,)
        Mean of ``gamma_hat`` across domains.
    tau_sq : ndarray of shape (n_domains,)
        Variance of ``gamma_hat`` across domains.
    a_prior : ndarray of shape (n_domains,)
        Shape parameters of the inverse-gamma prior.
    b_prior : ndarray of shape (n_domains,)
        Scale parameters of the inverse-gamma prior.
    """

    gamma_bar = gamma_hat.mean(1)
    tau_sq = gamma_hat.var(1, ddof=1)

    m = delta_hat.mean(1)
    s2 = delta_hat.var(1, ddof=1)

    a_prior = (2 * s2 + m**2) / s2
    b_prior = (m * s2 + m**3) / s2

    return gamma_bar, tau_sq, a_prior, b_prior


def _postmean(gamma_hat, gamma_bar, n, delta_star, tau_2):
    """Compute the posterior mean for the location parameters.

    Parameters
    ----------
    gamma_hat : ndarray of shape (n_domains, n_features)
        Least-squares estimates of the domain-specific location parameters.
    gamma_bar : ndarray of shape (n_domains,)
        Mean location parameters across domains.
    n : ndarray of shape (n_domains, n_features)
        Number of observations per domain and feature.
    delta_star : ndarray of shape (n_domains, n_features)
        Current estimate of the posterior scale parameters.
    tau_2 : ndarray of shape (n_domains,)
        Prior variances of the location parameters.

    Returns
    -------
    gamma_star : ndarray of shape (n_domains, n_features)
        Posterior mean of the location parameters.
    """

    tau_2 = tau_2.reshape(-1, 1)
    gamma_bar = gamma_bar.reshape(-1, 1)

    return (tau_2 * n * gamma_hat + delta_star * gamma_bar) / (tau_2 * n + delta_star)


def _postvar(ssr, n, a_prior, b_prior):
    """Compute the posterior variance for the scale parameters.

    Parameters
    ----------
    ssr : ndarray of shape (n_domains, n_features)
        Sum of squared residuals per feature and domain.
    n : ndarray of shape (n_domains, n_features)
        Number of observations per domain and feature.
    a_prior : ndarray of shape (n_domains,)
        Shape parameters of the inverse-gamma prior.
    b_prior : ndarray of shape (n_domains,)
        Scale parameters of the inverse-gamma prior.

    Returns
    -------
    delta_star : ndarray of shape (n_domains, n_features)
        Posterior variance estimates for each domain.
    """

    a_prior = a_prior.reshape(-1, 1)
    b_prior = b_prior.reshape(-1, 1)
    return (0.5 * ssr + b_prior) / (n / 2.0 + a_prior - 1.0)


class ComBat(OneToOneFeatureMixin, BaseAdapter):
    """Harmonize features across domains via the ComBat algorithm.

    The estimator implements a domain-adaptation variant of ComBat, combining
    empirical Bayes location and scale adjustments with optional covariates.
    Domains are encoded through metadata passed via ``domains`` and covariates
    through ``covariates``.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations for the empirical Bayes updates.
    tol : float, default=1e-4
        Tolerance for convergence of the empirical Bayes updates.
    copy : bool, default=True
        If ``False``, try to perform in-place operations when possible.
    """

    _parameter_constraints = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(self, max_iter=100, tol=0.0001, copy=True):
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def _fit_standardization_parameters(self, X, D, domains_counts):
        """Estimate grand mean and pooled variance used for standardization."""
        domains_ratio = domains_counts / domains_counts.sum()

        n_samples = X.shape[0]
        n_domains = len(self.domains_)

        # Do least square to find a linear transformation for the design
        # matrix to the original output
        beta_hat = la.lstsq(D, X)[0]
        # Estimate the weighted domain-wise mean
        grand_mean = np.dot(domains_ratio, beta_hat[:n_domains])
        # Estimate the pooled variance
        ssr = ((X - np.dot(D, beta_hat)) ** 2).sum(0)
        pooled_var = ssr / (n_samples - 1)

        self.beta_hat_ = beta_hat
        self.grand_mean_ = grand_mean
        self.pooled_var_ = pooled_var

    def _standardize(self, X, D):
        """Center and scale the data using fitted standardization parameters."""
        n_domains = len(self.domains_)
        pooled_std = np.sqrt(self.pooled_var_)

        # Mask non-domain columns
        D_covs = D.copy()
        D_covs[:, :n_domains] = 0

        # Transform design matrix to the feature space
        X_hat = np.dot(D_covs, self.beta_hat_)

        # Estimate the standardized mean
        X_bar = self.grand_mean_ + X_hat

        # Standardize the data
        X_std = (X - X_bar) / pooled_std

        return X_std, X_bar

    def _fit_ls_parameters(self, X, D, domains):
        """Fit least-squares estimates of domain-specific parameters."""
        n_domains = len(self.domains_)

        # Do least square to find a linear transformation
        # from the one-hot encoded domains to the original output
        D_domains = D[:, :n_domains]
        gamma_hat = la.lstsq(D_domains, X)[0]

        # Estimate the standardized (unbiased) domain variance
        delta_hat = [X[domains == domain].var(0, ddof=1) for domain in self.domains_]
        delta_hat = np.row_stack(delta_hat)

        return gamma_hat, delta_hat

    def _find_parametric_adjustments(
        self,
        X,
        domains,
        domains_counts,
        gamma_hat,
        delta_hat,
        gamma_bar,
        tau_sq,
        a_prior,
        b_prior,
    ):
        """Iteratively estimate empirical Bayes adjustments for each domain."""
        max_domain_samples = domains_counts.max()
        n_domains = len(self.domains_)
        d_features = X.shape[1]

        # Extend X to have the same shape as the design matrix
        # Intended to achieve more vectorization
        X_ext = np.full((max_domain_samples, n_domains, d_features), np.nan)
        for i, domain in enumerate(self.domains_):
            X_ext[: domains_counts[i], i] = X[domains == domain]

        # Number of samples per domain
        n = (~np.isnan(X_ext)).sum(0)

        gamma_hat_old = gamma_hat.copy()
        delta_hat_old = delta_hat.copy()

        # Empirical Bayes estimation
        # EM algorithm to find gamma_star and delta_star
        # gamma ~ N(gamma_star, delta_star), delta ~ IG(a_star, b_star)
        for i in range(self.max_iter):
            # E-step
            gamma_hat_new = _postmean(gamma_hat, gamma_bar, n, delta_hat_old, tau_sq)

            # M-step
            squared_residuals = (X_ext - gamma_hat_new) ** 2
            ssr = np.nansum(squared_residuals, 0)
            delta_hat_new = _postvar(ssr, n, a_prior, b_prior)

            close = np.allclose(gamma_hat_new, gamma_hat_old, self.tol, self.tol)
            close &= np.allclose(delta_hat_new, delta_hat_old, self.tol, self.tol)

            if close:
                break

            gamma_hat_old = gamma_hat_new
            delta_hat_old = delta_hat_new

        self.n_iter_ = i + 1
        self.gamma_star_ = gamma_hat_new
        self.delta_star_ = delta_hat_new

    def _adjust(self, X, D, X_bar, domains):
        """Apply empirical Bayes adjustments and reverse standardization."""
        n_domains = len(self.domains_)
        pooled_std = np.sqrt(self.pooled_var_)
        delta_star_sqrt = np.sqrt(self.delta_star_)

        X_bayes = X.copy()
        D_domains = D[:, :n_domains]

        for i, domain in enumerate(self.domains_):
            domain_mask = domain == domains

            if domain_mask.sum() < 1:
                continue

            X_bayes_i = X_bayes[domain_mask]
            D_domains_i = D_domains[domain_mask]

            residuals = X_bayes_i - np.dot(D_domains_i, self.gamma_star_)
            X_bayes[domain_mask] = residuals / delta_star_sqrt[i]

        X_bayes = X_bayes * pooled_std + X_bar

        return X_bayes

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, domains=None, covariates=None):
        """Estimate ComBat harmonization parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API compatibility.
        domains : array-like of shape (n_samples,)
            Domain labels for each sample. Required.
        covariates : array-like of shape (n_samples, n_covariates), default=None
            Optional covariates to regress out during harmonization.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate (and cast) input data
        X = validate_data(self, X, copy=self.copy)
        domains, domains_ohe = self._validate_domains(domains, X, required=True, return_value="both")
        covariates = self._validate_covariates(covariates, X, copy=self.copy)
        self.requires_cov_ = covariates is not None

        # Make design matrix
        D = self._augment(domains=domains_ohe, covariates=covariates)
        _, domains_counts = np.unique(domains, return_counts=True)

        # Fit standardization parameters
        self._fit_standardization_parameters(X, D, domains_counts)

        # Standardize data
        X_std, _ = self._standardize(X, D)

        # Fit L/S parameters
        gamma_hat, delta_hat = self._fit_ls_parameters(X_std, D, domains)

        # Estimate empirical Bayes priors
        gamma_bar, tau_sq, a_prior, b_prior = _estimate_priors(gamma_hat, delta_hat)

        # Find parametric adjustments
        self._find_parametric_adjustments(
            X_std,
            domains,
            domains_counts,
            gamma_hat,
            delta_hat,
            gamma_bar,
            tau_sq,
            a_prior,
            b_prior,
        )

        return self

    def transform(self, X, domains=None, covariates=None):
        """Harmonize new samples using the fitted parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to harmonize.
        domains : array-like of shape (n_samples,)
            Domain labels for each sample. Required.
        covariates : array-like of shape (n_samples, n_covariates), default=None
            Covariates to align with the fitted model. Required when used at
            fit time.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Harmonized data.
        """
        check_is_fitted(self)
        # Validate (and cast) input data
        X = validate_data(self, X, copy=self.copy)
        domains, domains_ohe = self._validate_domains(
            domains,
            X,
            reset=False,
            required=True,
            return_value="both",
            copy=self.copy,
        )
        covariates = self._validate_covariates(covariates, X, required=self.requires_cov_, copy=self.copy)

        # Make design matrix
        D = self._augment(domains=domains_ohe, covariates=covariates)

        # Standardize data
        X_std, X_bar = self._standardize(X, D)

        # Adjust data
        X_bayes = self._adjust(X_std, D, X_bar, domains)

        return X_bayes
