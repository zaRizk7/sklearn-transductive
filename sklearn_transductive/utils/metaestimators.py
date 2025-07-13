__all__ = ["estimator_has", "subestimator_has", "estimator_attr_true"]


def estimator_has(attr, ensure_all=False):
    def check(self):
        attr_ = attr
        if isinstance(attr, str):
            attr_ = (attr_,)

        agg = all if ensure_all else any

        return agg(hasattr(self, a) for a in attr_)

    return check


def subestimator_has(subestimator, attr, ensure_all=False):
    def check(self):
        subestimator_ = subestimator
        if isinstance(subestimator, str):
            subestimator_ = (subestimator_,)

        attr_ = attr
        if isinstance(attr_, str):
            attr_ = (attr_,)

        agg = all if ensure_all else any

        return agg(hasattr(getattr(self, s), a) for s in subestimator_ for a in attr_)

    return check


def estimator_attr_true(attr, ensure_all=False):
    def check(self):
        attr_ = attr
        if isinstance(attr_, str):
            attr_ = (attr_,)

        conds = []
        for a in attr_:
            a = getattr(self, a)
            if not isinstance(a, bool):
                msg = f"attribute '{attr}' is not a boolean!"
                raise ValueError(msg)

            conds.append(a)

        agg = all if ensure_all else any

        return agg(conds)

    return check
