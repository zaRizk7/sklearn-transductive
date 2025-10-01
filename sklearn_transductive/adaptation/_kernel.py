from numbers import Integral, Real

import numpy as np
import numpy.linalg as la
from scipy.linalg import eigh, inv
from scipy.sparse.linalg import eigsh
from sklearn.base import ClassNamePrefixFeaturesOutMixin
from sklearn.linear_model._ridge import _solve_cholesky_kernel
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.preprocessing import KernelCenterer, OneHotEncoder
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import _randomized_eigsh, safe_sparse_dot, svd_flip
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    _check_psd_eigenvalues,
    check_is_fitted,
    validate_data,
)

from ..utils.extmath import (
    centering_kernel,
    remove_significant_negative_eigenvalues,
    sort_eigencomponents,
)
from ..utils.metaestimators import estimator_attr_true
from ._base import BaseAdapter

__all__ = ["BaseKernelEigenAdapter", "MIDA", "TCA"]


# Basically an extension of KernelPCA when scale_components=True.
# The main distinction is the accommodation of domains and covariates
# and more customizable to allow it to be easily extended for
# various kernel-based adaptation methods.
class BaseKernelEigenAdapter(ClassNamePrefixFeaturesOutMixin, BaseAdapter):
    """Base class for kernel eigen-decomposition domain adapters.

    The estimator performs an eigendecomposition of a kernel matrix to produce
    domain-invariant components. Subclasses can customize the optimization
    objective by overriding :meth:`_make_solution_kernel`.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to retain. If ``None``, use all available
        components.
    ignore_y : bool, default=True
        If ``True``, the supervised signal ``y`` is ignored during fitting.
    augment : bool, default=False
        If ``True``, augment the input features with domains and covariates
        metadata prior to computing kernels.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'} or callable, default='linear'
        Kernel function to use. See :func:`~sklearn.metrics.pairwise.pairwise_kernels`
        for details.
    gamma : float or None, default=None
        Kernel coefficient for ``'rbf'``, ``'poly'`` and ``'sigmoid'`` kernels.
    degree : float, default=3
        Degree for the ``'poly'`` kernel.
    coef0 : float, default=1
        Independent term in ``'poly'`` and ``'sigmoid'`` kernels.
    kernel_params : dict or None, default=None
        Additional parameters to pass to kernel function when ``kernel`` is a
        callable.
    alpha : float, default=1.0
        Regularization parameter used when computing the inverse transform.
    fit_inverse_transform : bool, default=False
        Whether to compute the dual coefficients to perform an inverse
        transformation.
    eigen_solver : {'auto', 'dense', 'arpack', 'randomized'}, default='auto'
        Solver used for the eigendecomposition.
    tol : float, default=0
        Convergence tolerance for iterative solvers.
    max_iter : int or None, default=None
        Maximum number of iterations for the ARPACK solver.
    iterated_power : int or {'auto'}, default='auto'
        Number of power iterations when using the randomized solver.
    remove_zero_eig : bool, default=False
        If ``True``, remove eigencomponents with zero eigenvalues before
        returning the embedding.
    scale_components : bool, default=False
        If ``True``, scale eigenvectors by the inverse square root of the
        eigenvalues.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the ARPACK and randomized solvers.
    copy : bool, default=True
        If ``False``, try to avoid copying the input data when possible.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel for pairwise kernel evaluations.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "ignore_y": ["boolean"],
        "augment": ["boolean"],
        "kernel": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS) | {"precomputed"}),
            callable,
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "fit_inverse_transform": ["boolean"],
        "eigen_solver": [StrOptions({"auto", "dense", "arpack", "randomized"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "iterated_power": [
            Interval(Integral, 0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "remove_zero_eig": ["boolean"],
        "scale_components": ["boolean"],
        "random_state": ["random_state"],
        "copy": ["boolean"],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        n_components=None,
        *,
        ignore_y=True,
        augment=False,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        n_jobs=None,
    ):
        # Truncation parameters
        self.n_components = n_components

        # Supervision parameters
        self.ignore_y = ignore_y

        # Kernel parameters
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.copy = copy
        self.n_jobs = n_jobs

        # Eigendecomposition parameters
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state

        # Transform parameters
        self.scale_components = scale_components

        # Inverse transform parameters
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform

        # Additional adaptation parameters
        self.augment = augment

    @property
    def requires_domains_(self):
        return self.fit_has_domains_ and self.augment

    @property
    def requires_covs_(self):
        return self.fit_has_covs_ and self.augment

    @property
    def _n_features_out(self):
        return self.eigenvalues_.shape[0]

    def _make_solution_kernel(self, K, y=None, domains=None, covariates=None):
        """Return the kernel used to solve the eigenproblem.

        Subclasses can override this hook to incorporate additional terms
        derived from labels, domains, or covariates before the eigensolver is
        called.

        Parameters
        ----------
        K : ndarray or tuple of ndarray
            Centered kernel matrix, or a pair ``(A, B)`` for generalized
            eigenproblems.
        y : array-like of shape (n_samples, n_targets), default=None
            Optional supervision signal for subclasses that leverage labels.
        domains : array-like, default=None
            Encoded domain metadata aligned with the rows of ``K``.
        covariates : array-like, default=None
            Additional metadata aligned with the rows of ``K``.

        Returns
        -------
        ndarray or tuple of ndarray
            Kernel passed to :meth:`_fit_transform_weights`.
        """
        return K

    def _more_tags(self):
        """Return estimator tags communicated to scikit-learn utilities."""
        return {
            "pairwise": self.kernel == "precomputed",
            "_xfail_checks": {"check_transformer_n_iter": "Follows similar implementation to KernelPCA."},
        }

    def _get_kernel(self, X, Y=None):
        """Compute the pairwise kernel matrix between ``X`` and ``Y``.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Left argument for the kernel evaluation.
        Y : array-like of shape (n_samples_Y, n_features), default=None
            Right argument for the kernel evaluation. When ``None``, use
            ``X`` for a self-kernel.

        Returns
        -------
        ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        """
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params)

    def _estimate_n_components(self, K):
        """Determine how many eigencomponents to retain.

        Parameters
        ----------
        K : ndarray or tuple of ndarray
            Kernel matrix (or matrices) describing the eigenproblem.

        Returns
        -------
        n_components : int
            Number of components to compute based on ``self.n_components``
            and the size of ``K``.
        """
        if isinstance(K, (tuple, list)):
            n_components = K[0].shape[0]
        else:
            n_components = K.shape[0]
        if self.n_components is not None:
            n_components = min(n_components, self.n_components)
        return n_components

    def _get_solver(self, K, n_components):
        """Select an eigensolver strategy based on the kernel size.

        Parameters
        ----------
        K : ndarray or tuple of ndarray
            Kernel matrix (or matrices) describing the eigenproblem.
        n_components : int
            Number of components requested. Included for API symmetry with
            subclasses that may use it.

        Returns
        -------
        solver : {'dense', 'arpack', 'randomized'}
            Name of the solver to be used by :meth:`_eigendecompose`.
        """
        if isinstance(K, (tuple, list)):
            K_size = K[0].shape[0]
        else:
            K_size = K.shape[0]

        solver = self.eigen_solver
        if solver == "auto" and K_size > 200 and K_size < 10:
            solver = "arpack"
        elif solver == "auto":
            solver = "dense"
        return solver

    def _eigendecompose(self, K, n_components, solver):
        """Compute eigenvalues and eigenvectors for the solution kernel.

        Parameters
        ----------
        K : ndarray or tuple of ndarray
            Matrix defining the eigenvalue problem. When provided as a tuple,
            it is interpreted as ``(A, B)`` for a generalized problem.
        n_components : int
            Number of eigenpairs to compute.
        solver : {'dense', 'arpack', 'randomized'}
            Solver strategy selected by :meth:`_get_solver`.

        Returns
        -------
        eigenvalues : ndarray of shape (n_components,)
            Selected eigenvalues sorted in ascending order.
        eigenvectors : ndarray of shape (n_samples, n_components)
            Corresponding eigenvectors.
        """
        # is a generalized eigenvalue problem
        if isinstance(K, (tuple, list)):
            A, B = K
        else:
            A, B = K, None

        if solver == "arpack":
            v0 = _init_arpack_v0(A.shape[0], self.random_state)
            return eigsh(
                A,
                n_components,
                B,
                which="LA",
                tol=self.tol,
                maxiter=self.max_iter,
                v0=v0,
            )

        if solver == "randomized":
            if B is not None:
                A = inv(B) @ A

            return _randomized_eigsh(
                A,
                n_components=n_components,
                n_iter=self.iterated_power,
                random_state=self.random_state,
                selection="module",
            )

        # If solver is 'dense', use standard scipy.linalg.eigh
        # Note: subset_by_index specifies the indices of smallest/largest to return
        index = (A.shape[0] - n_components, A.shape[0] - 1)
        return eigh(A, B, subset_by_index=index)

    def _remove_zero_eigencomponents(self, eigenvalues, eigenvectors):
        """Filter out zero eigenvalues and associated eigenvectors.

        Parameters
        ----------
        eigenvalues : ndarray of shape (n_components,)
            Eigenvalues obtained from the solver.
        eigenvectors : ndarray of shape (n_samples, n_components)
            Eigenvectors corresponding to ``eigenvalues``.

        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues with zero entries removed when requested.
        eigenvectors : ndarray
            Eigenvectors filtered to match the returned eigenvalues.
        """
        if self.remove_zero_eig or self.n_components is None:
            pos_mask = eigenvalues > 0
            eigenvectors = eigenvectors[:, pos_mask]
            eigenvalues = eigenvalues[pos_mask]

        return eigenvalues, eigenvectors

    @property
    def _eigen_postprocess_steps(self):
        """Ordered post-processing steps applied to eigenpairs."""
        # Please override to modify the postprocessing steps
        # NOTE: The step can only be removed or reordered.
        return (
            "remove_significant_negative_eigenvalues",
            "check_psd_eigenvalues",
            "svd_flip",
            "sort_eigencomponents",
            "remove_zero_eigencomponents",
        )

    def _postprocess_eig(self, eigenvalues, eigenvectors):
        """Apply numerical post-processing to the eigendecomposition.

        Parameters
        ----------
        eigenvalues : ndarray of shape (n_components,)
            Raw eigenvalues returned by the solver.
        eigenvectors : ndarray of shape (n_samples, n_components)
            Raw eigenvectors returned by the solver.

        Returns
        -------
        eigenvalues : ndarray
            Post-processed eigenvalues after applying the configured steps.
        eigenvectors : ndarray
            Post-processed eigenvectors aligned with ``eigenvalues``.
        """
        for step in self._eigen_postprocess_steps:
            if step == "remove_significant_negative_eigenvalues":
                eigenvalues = remove_significant_negative_eigenvalues(eigenvalues)
            if step == "check_psd_eigenvalues":
                eigenvalues = _check_psd_eigenvalues(eigenvalues)
            if step == "svd_flip":
                eigenvectors, _ = svd_flip(eigenvectors, None)
            if step == "sort_eigencomponents":
                eigenvalues, eigenvectors = sort_eigencomponents(eigenvalues, eigenvectors)
            if step == "remove_zero_eigencomponents":
                eigenvalues, eigenvectors = self._remove_zero_eigencomponents(eigenvalues, eigenvectors)

        return eigenvalues, eigenvectors

    def _fit_transform_weights(self, K):
        """Fit eigenbasis used to project data into the adapted subspace.

        Parameters
        ----------
        K : ndarray or tuple of ndarray
            Solution kernel returned by :meth:`_make_solution_kernel`.
        """
        n_components = self._estimate_n_components(K)
        solver = self._get_solver(K, n_components)

        eigenvalues, eigenvectors = self._eigendecompose(K, n_components, solver)

        eigenvalues, eigenvectors = self._postprocess_eig(eigenvalues, eigenvectors)

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

    def _fit_inverse_transform_weights(self):
        """Estimate dual coefficients required for inverse transformations.

        The dual system is solved in the transformed space so that
        :meth:`inverse_transform` can map projections back to the original
        feature space when ``fit_inverse_transform`` is enabled.
        """
        if not self.fit_inverse_transform:
            return

        if self.kernel == "precomputed" and self.fit_inverse_transform:
            msg = "Cannot fit the inverse transform when kernel='precomputed'."
            raise ValueError(msg)

        self.Z_fit_ = self.transform(self.X_fit_)
        K_z = self._get_kernel(self.Z_fit_)

        self.dual_coef_ = _solve_cholesky_kernel(K_z, self.X_fit_, self.alpha)

    def _scale_eigenvectors(self):
        """Return eigenvectors optionally rescaled by the eigenvalues.

        Returns
        -------
        ndarray of shape (n_samples_train, n_components)
            Eigenvectors scaled according to ``self.scale_components``.
        """
        W = self.eigenvectors_

        if not self.scale_components:
            return W

        s2 = self.eigenvalues_
        s = np.sqrt(s2)

        non_zeros = np.flatnonzero(s)
        W_hat = np.zeros_like(W)
        W_hat[:, non_zeros] = W[:, non_zeros] / s[non_zeros]

        return W_hat

    # TODO: Add weight to the original space
    @property
    def orig_coef_(self):
        """Coefficients of the linear transformation in the original space."""
        check_is_fitted(self)
        if self.kernel != "linear":
            raise NotImplementedError("Supports linear kernel only.")
        # X_fit_ shape: (n_samples_train, n_features)
        # W shape: (n_samples_train, n_components)
        W = self._scale_eigenvectors()
        # W_orig shape: (n_components, n_features)
        return safe_sparse_dot(self.X_fit_.T, W)

    def fit(self, X, y=None, domains=None, covariates=None, **fit_params):
        """Fit the adapter by solving a kernel eigenvalue problem.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,), default=None
            Optional supervision signal. Used when ``ignore_y`` is ``False``.
        domains : array-like of shape (n_samples,), default=None
            Domain labels for each input sample.
        covariates : array-like of shape (n_samples, n_covariates), default=None
            Additional metadata whose effect should be removed during
            adaptation.
        **fit_params : dict
            Additional fit parameters. They are currently ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        check_params = {
            "accept_sparse": False if self.fit_inverse_transform else "csr",
            "copy": self.copy,
        }
        if y is None or self.ignore_y:
            X = validate_data(self, X, **check_params)
            y_ohe = np.zeros((X.shape[0], 1))
        else:
            X, y = validate_data(self, X, y, **check_params)

            drop = (-1,) if np.any(y == -1) else None
            ohe = OneHotEncoder(sparse_output=False, drop=drop)
            y_ohe = ohe.fit_transform(y.reshape(-1, 1))

        self.fit_has_domains_ = domains is not None
        self.fit_has_covs_ = covariates is not None

        domains = self._validate_domains(domains, X, required=self.requires_domains_, copy=self.copy)
        covariates = self._validate_covariates(covariates, X, required=self.requires_covs_, copy=self.copy)

        X = self._augment(X, domains, covariates) if self.augment else X
        self.X_fit_ = X
        self.gamma_ = 1 / X.shape[1] if self.gamma is None else self.gamma

        K = self._get_kernel(X)
        self._centerer = KernelCenterer()
        K_c = self._centerer.fit_transform(K)

        A = self._make_solution_kernel(K_c, y_ohe, domains, covariates)

        self._fit_transform_weights(A)
        self._fit_inverse_transform_weights()

        return self

    def transform(self, X, domains=None, covariates=None):
        """Project samples onto the domain-adapted components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        domains : array-like of shape (n_samples,), default=None
            Domain labels. Required when ``augment`` is ``True`` and domains
            were provided at fit time.
        covariates : array-like of shape (n_samples, n_covariates), default=None
            Additional metadata required when ``augment`` is ``True`` and
            covariates were provided during fitting.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedded representation of the input samples.
        """
        check_is_fitted(self)
        accept_sparse = False if self.fit_inverse_transform else "csr"
        X = validate_data(self, X, accept_sparse=accept_sparse, reset=False)
        domains = self._validate_domains(domains, X, False, self.requires_domains_)
        covariates = self._validate_covariates(covariates, X, False, self.requires_covs_)

        X = self._augment(X, domains, covariates) if self.augment else X
        K = self._get_kernel(X, self.X_fit_)
        K_c = self._centerer.transform(K)

        W = self._scale_eigenvectors()

        return safe_sparse_dot(K_c, W)

    @available_if(estimator_attr_true("fit_inverse_transform"))
    def inverse_transform(self, Z):
        """Map embedded samples back to the original feature space.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)
            Embedded representation of the samples.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed samples in the original feature space.
        """
        check_is_fitted(self)
        K_z = self._get_kernel(Z, self.Z_fit_)
        return safe_sparse_dot(K_z, self.dual_coef_)


class MIDA(BaseKernelEigenAdapter):
    """Maximum Independence Domain Adaptation (MIDA).

    Minimizes the dependence between transformed features, domains and class
    labels by maximizing independence in Hilbert space while preserving
    discriminative structure.

    Parameters
    ----------
    mu : float, default=1.0
        Regularization parameter applied to the centering term.
    eta : float, default=1.0
        Regularization strength for the class dependence term.

    See Also
    --------
    BaseKernelEigenAdapter
    """

    _parameter_constraints = {
        **BaseKernelEigenAdapter._parameter_constraints,
        "mu": [Interval(Real, 0, None, closed="neither")],
        "eta": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components=None,
        *,
        mu=1.0,
        eta=1.0,
        ignore_y=False,
        augment=False,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        n_jobs=None,
    ):
        # MIDA parameters
        self.mu = mu  # L2 kernel regularization parameter
        self.eta = eta  # Class dependency regularization parameter

        # Kernel and Eigendecomposition parameters
        super().__init__(
            n_components,
            ignore_y=ignore_y,
            augment=augment,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            scale_components=scale_components,
            random_state=random_state,
            copy=copy,
            n_jobs=n_jobs,
        )

    def _make_solution_kernel(self, K, y=None, domains=None, covariates=None):
        """Construct the MIDA objective kernel incorporating metadata.

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Centered kernel matrix of the training data.
        y : ndarray of shape (n_samples, n_targets)
            One-hot encoded supervision signal.
        domains : ndarray of shape (n_samples, n_domain_features), default=None
            Encoded domains for each sample.
        covariates : ndarray of shape (n_samples, n_covariates), default=None
            Additional covariates aligned with the samples.

        Returns
        -------
        ndarray of shape (n_samples, n_samples)
            Solution matrix whose eigenvectors define the adapted subspace.
        """
        if domains is None:
            domains = np.zeros((K.shape[0], 1))

        if covariates is None:
            covariates = np.zeros((K.shape[0], 1))

        D = self._augment(domains=domains, covariates=covariates)

        H = centering_kernel(K.shape[0], dtype=K.dtype)
        K_y = pairwise_kernels(y, n_jobs=self.n_jobs)
        K_d = pairwise_kernels(D, n_jobs=self.n_jobs)

        centerer = KernelCenterer()
        K_y = centerer.fit_transform(K_y)
        K_d = centerer.fit_transform(K_d)

        A = la.multi_dot((K, self.mu * H + self.eta * K_y - K_d, K))

        return A


class TCA(BaseKernelEigenAdapter):
    """Transfer Component Analysis (TCA).

    Learns a kernel subspace that minimizes domain shift by aligning
    distributions while retaining data variance.

    Parameters
    ----------
    mu : float, default=0.1
        Trade-off parameter balancing alignment and variance preservation.

    See Also
    --------
    BaseKernelEigenAdapter
    """

    _parameter_constraints = {
        **BaseKernelEigenAdapter._parameter_constraints,
        "mu": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components=None,
        *,
        mu=0.1,
        ignore_y=False,
        augment=False,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        n_jobs=None,
    ):
        # TCA parameters
        self.mu = mu  # Trade-off parameter?

        # Kernel and Eigendecomposition parameters
        super().__init__(
            n_components,
            ignore_y=ignore_y,
            augment=augment,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            scale_components=scale_components,
            random_state=random_state,
            copy=copy,
            n_jobs=n_jobs,
        )

    def _make_solution_kernel(self, K, y=None, domains=None, covariates=None):
        """Construct the TCA generalized eigenproblem matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Centered kernel matrix of the training data.
        y : Ignored
            Present for API compatibility; TCA does not use supervision.
        domains : ndarray of shape (n_samples, n_domain_features), default=None
            Encoded domains for each sample.
        covariates : Ignored
            TCA does not currently leverage covariate information.

        Returns
        -------
        ndarray of shape (n_samples, n_samples)
            Matrix representing the generalized eigenproblem solved by TCA.
        """
        if domains is None:
            # Assume all samples are from the same domain
            domains = np.ones((K.shape[0], 1))

        # Ignore covariates (technically can be concatenated to K as
        # used in the parent class, but not necessary for TCA)
        G = domains.sum(0)  # (n_domains)
        G = (G * domains).sum(1, keepdims=True)  # (n_samples, 1)

        # Get original domain indices, waste of computation
        # but validated with OneHotEncoder anyway
        # 0 if x_i, x_j not in the same domain, 1 otherwise
        L = pairwise_kernels(domains, n_jobs=self.n_jobs)
        # Rescale such that L denotes the sign of L_ij
        # Get sign, '+' if X_i and X_j are in the same domain, '-' otherwise
        L = L * 2 - 1
        # -1 / (n_source * n_target) if x_i, x_j not in the same domain
        # 1 / (n_source^2) if x_i, x_j in the same domain
        L = L / pairwise_kernels(G, n_jobs=self.n_jobs)
        # NOTE: This is as described in the paper, but the implementation
        #       allows to generalize to multiple domains rather than just
        #       two domains, which is the source and target domains.

        # Identity matrix
        I = np.eye(K.shape[0])  # noqa: E741
        # Centering matrix
        H = centering_kernel(K.shape[0], K.dtype)

        # Computes the solution matrix, (I+mu*KLK)^-1 KHK
        A = la.multi_dot((K, H, K))
        B = self.mu * I + la.multi_dot((K, L, K))

        return inv(B) @ A
