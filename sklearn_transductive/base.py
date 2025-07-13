import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array, check_consistent_length, safe_mask
from sklearn.utils.validation import _num_features

__all__ = ["DomainValidationMixin", "CovariateValidationMixin"]

DOMAIN_VALIDATION_RETURNS = ("onehot", "validated", "both")


def _check_group_return_value(return_value):
    if return_value not in DOMAIN_VALIDATION_RETURNS:
        msg = f"return_value must be one of {DOMAIN_VALIDATION_RETURNS}. "
        msg += f"Got return_value='{return_value}'"
        raise ValueError(msg)


def _check_required(value, required, input_name="value"):
    if required and value is None:
        msg = f"{input_name} must be provided when required=True"
        raise ValueError(msg)


class DomainValidationMixin:
    @property
    def domains_(self):
        return self._ge.categories_[0]

    @property
    def n_domains_(self):
        return len(self.domains_)

    def _validate_domains(
        self,
        domains,
        X,
        reset=True,
        required=False,
        return_value="onehot",
        **check_params,
    ):
        _check_group_return_value(return_value)
        _check_required(domains, required, "domains")

        if domains is not None:
            check_consistent_length(domains, X)
            default_check_params = {
                "dtype": None,
                "ensure_2d": False,
                "estimator": self,
                "input_name": "domains",
            }

            check_params = {**default_check_params, **check_params}
            domains = check_array(domains, **check_params)
            domains = domains.reshape(-1, 1)

        if domains is not None and reset:
            self._ge = OneHotEncoder(sparse_output=False).fit(domains)

        if domains is not None and return_value in ("onehot", "both"):
            domains_ohe = self._ge.transform(domains)

        if domains is not None and return_value == "onehot":
            return domains_ohe

        if domains is not None:
            domains = domains.ravel()

        if domains is not None and return_value == "both":
            return domains, domains_ohe

        return domains


class CovariateValidationMixin:
    def _validate_covariates(
        self, covariates, X, reset=True, required=False, **check_params
    ):
        _check_required(covariates, required, "covariates")

        if covariates is not None:
            check_consistent_length(covariates, X)
            default_check_params = {"estimator": self, "input_name": "covariates"}
            check_params = {**default_check_params, **check_params}
            covariates = check_array(covariates, **check_params)

        if required and check_params.get("ensure_2d", True):
            self._check_n_covariates(covariates, reset)

        return covariates

    def _check_n_covariates(self, covariates, reset):
        # Taken directly from _check_n_features from BaseEstimator
        # modified to check the number of covariates
        try:
            n_covariates = _num_features(covariates)
        except TypeError as e:
            if not reset and hasattr(self, "n_covariates_"):
                raise ValueError(
                    "covariates does not contain any values, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_covariates_} covariates"
                ) from e
            # If the number of covariates is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_covariates_ = n_covariates

        if not hasattr(self, "n_covariates_"):
            # Skip this check if the expected number of expected input covariates
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_covariates != self.n_covariates_:
            raise ValueError(
                f"covariates has {n_covariates} covariates, but {self.__class__.__name__} "
                f"is expecting {self.n_covariates_} covariates as input."
            )
