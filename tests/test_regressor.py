import pytest
import os 

import pandas as pd 

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

from plotly.graph_objs import Figure

from rule_estimator import *


@pytest.fixture
def model():
    return RuleRegressor(
        LesserThanNode(
            col='s5',
            cutoff=-0.003,
            if_true=LesserThan(
                col='bmi',
                cutoff=0.006,
                prediction=96,
                default=161
            ),
            if_false=LesserThan(
                col='bmi',
                cutoff=0.014,
                prediction=162,
                default=225
            ),
        )
    )

@pytest.fixture
def data():
    return load_diabetes(return_X_y=True, as_frame=True)


def test_rule_regressor(model, data):
    X, y = data
    assert mean_squared_error(y, model.predict(X), squared=False) < 60

def test_reg_score_rules(model, data):
    X, y = data
    assert isinstance(model.score_rules(X, y), pd.DataFrame)

def test_reg_model_suggest(model, data):
    X, y = data
    assert model.suggest_rule(0, X, y).startswith("LesserThan(col='s5', cutoff=-0.003")
    assert model.suggest_rule(0, X, y, kind='prediction').startswith("PredictionRule(prediction=152")
    assert model.suggest_rule(0, X, y, kind='node').startswith("LesserThanNode(col='s5', cutoff=-0.003")

    assert model.suggest_rule(2, X, y).startswith("LesserThan(col='bmi', cutoff=0.014")
    assert model.suggest_rule(2, X, y, kind='prediction').startswith("PredictionRule(prediction=193")
    assert model.suggest_rule(2, X, y, kind='node').startswith("LesserThanNode(col='bmi', cutoff=0.014")
