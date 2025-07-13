# sklearn-transductive
An rewrite of some sklearn functionalities to allow transductive learning to do domain adaptation.

Most of the contents in the code are copy and pasted from the original `scikit-learn` code, with added functionality like overriding `OptunaSearchCV` from `optuna` to allow Bayesian hyperparameter search.