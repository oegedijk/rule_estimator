import pytest
import os 

import pandas as pd 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from plotly.graph_objs import Figure

from rule_estimator import *

@pytest.fixture
def model():
    return RuleClassifier(
        LesserThanNode("petal length (cm)", 1.91, # BinaryDecisionNode
            if_true=PredictionRule(prediction=0), # PredictionRule: always predict 0
            if_false=CaseWhen([
                        # Go through these rules and if one applies, assign the prediction
                        LesserThan("petal length (cm)", 4.5, prediction=1),
                        GreaterThan("petal length (cm)", 5.1, prediction=2),
                        LesserThan("petal width (cm)", 1.4, prediction=1),
                        GreaterThan("petal width (cm)", 1.8, prediction=2),
                    ], default=1 # if no rule applies, assign prediction=1
                ),
        ),   
    )

@pytest.fixture
def model_with_estimator():
    return RuleClassifier(
        LesserThanNode("petal length (cm)", 1.9, 
                if_true=PredictionRule(0), 
                if_false=CaseWhen([
                        LesserThan("petal length (cm)", 4.5, 1),
                        GreaterThan("petal length (cm)", 5.1, 2),
                        LesserThan("petal width (cm)", 1.4, 1),
                        GreaterThan("petal width (cm)", 1.8, 2),
                    ]),
        ), 
        final_estimator=DecisionTreeClassifier(),
        fit_remaining_only=False
    )

@pytest.fixture
def data():
    return load_iris(return_X_y=True, as_frame=True)

def test_model_to_yaml(model):
    assert isinstance(model.to_yaml(), str)

def test_store_model_to_yaml(model):
    if os.path.exists("modeltest.yaml"):
        os.remove("modeltest.yaml")
    
    model.to_yaml("modeltest.yaml")
    
    assert os.path.exists("modeltest.yaml")
    os.remove("modeltest.yaml")


def test_load_model_from_yaml(model):
    if os.path.exists("modeltest.yaml"):
        os.remove("modeltest.yaml")
    
    model.to_yaml("modeltest.yaml")
    assert os.path.exists("modeltest.yaml")
    
    model2 = RuleClassifier.from_yaml("modeltest.yaml")
    assert isinstance(model2, RuleClassifier)
    os.remove("modeltest.yaml")

def test_model_to_code(model):
    assert model.to_code().startswith("\nRuleClassifier")

def test_rule_describe(model):
    assert model.describe().startswith("RuleClassifier")

def test_final_estimator(model_with_estimator, data):
    X, y = data
    model_with_estimator.fit(X, y)
    accuracy_score(y, model_with_estimator.predict(X)) > 0.98

def test_rule_classifier(model, data):
    X, y = data
    assert accuracy_score(y, model.predict(X)) > 0.9

def test_model_plot(model):
    assert isinstance(model.plot(), Figure)

def test_model_plot(model, data):
    X, y = data
    assert isinstance(model.parallel_coordinates(X, y), Figure)

def test_score_rules(model, data):
    X, y = data
    assert isinstance(model.score_rules(X, y), pd.DataFrame)

def test_get_rule_input(model, data):
    X, y = data
    input_X, input_y = model.get_rule_input(4, X, y)
    assert input_X.shape == (71, 4)
    assert input_y.shape == (71,)

def test_get_rule_leftover(model, data):
    X, y = data
    input_X, input_y = model.get_rule_leftover(4, X, y)
    assert input_X.shape == (37, 4)
    assert input_y.shape == (37,)

def test_model_suggest(model, data):
    X, y = data
    assert model.suggest_rule(6, X, y).startswith("LesserThan(col='petal width (cm)', cutoff=1.75")

def test_model_suggest(model, data):
    X, y = data
    assert model.suggest_rule(6, X, y, after=True).startswith("LesserThan(col='petal width (cm)', cutoff=1.65")

def test_model_get_params(model):
    params = model.get_rule_params(3)
    assert isinstance(params, dict)
    assert params['cutoff'] == 4.5

def test_model_set_params(model):
    model.set_rule_params(3, cutoff=4.6)
    params = model.get_rule_params(3)
    assert isinstance(params, dict)
    assert params['cutoff'] == 4.6

def test_get_rule(model):
    assert isinstance(model.get_rule(5), BusinessRule)
    assert isinstance(model.get_rule(5), LesserThan)

    rule = model.get_rule(5)
    params = rule._stored_params
    assert params['cutoff'] == 1.6

def test_get_rule(model):
    model.replace_rule(5, LesserThan(col='petal width (cm)', cutoff=1.5, prediction=1))

    assert isinstance(model.get_rule(5), BusinessRule)
    assert isinstance(model.get_rule(5), LesserThan)
    
    rule = model.get_rule(5)
    params = rule._stored_params
    assert params['cutoff'] == 1.5











