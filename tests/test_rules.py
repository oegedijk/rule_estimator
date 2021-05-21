import os 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from rule_estimator import *


def test_rule_classifier():
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = RuleClassifier(
        LesserThanNode("petal length (cm)", 1.9, 
                if_true=DummyRule(0), 
                if_false=CaseWhen([
                        LesserThan("petal length (cm)", 4.5, 1),
                        GreaterThan("petal length (cm)", 5.1, 2),
                        LesserThan("petal width (cm)", 1.4, 1),
                        GreaterThan("petal width (cm)", 1.8, 2),
                    ], default=1),
        ), 
        default=2
    )

    assert accuracy_score(y, model.predict(X)) > 0.9


def test_model_to_yaml():
    model = RuleClassifier(
        LesserThanNode("petal length (cm)", 1.9, 
                if_true=DummyRule(0), 
                if_false=CaseWhen([
                        LesserThan("petal length (cm)", 4.5, 1),
                        GreaterThan("petal length (cm)", 5.1, 2),
                        LesserThan("petal width (cm)", 1.4, 1),
                        GreaterThan("petal width (cm)", 1.8, 2),
                    ], default=1),
        ), 
        default=2
    )
    assert isinstance(model.to_yaml(), str)


def test_store_model_to_yaml():
    model = RuleClassifier(
        LesserThanNode("petal length (cm)", 1.9, 
                if_true=DummyRule(0), 
                if_false=CaseWhen([
                        LesserThan("petal length (cm)", 4.5, 1),
                        GreaterThan("petal length (cm)", 5.1, 2),
                        LesserThan("petal width (cm)", 1.4, 1),
                        GreaterThan("petal width (cm)", 1.8, 2),
                    ], default=1),
        ), 
        default=2
    )

    if os.path.exists("modeltest.yaml"):
        os.remove("modeltest.yaml")
    model.to_yaml("modeltest.yaml")
    assert os.path.exists("modeltest.yaml")
    os.remove("modeltest.yaml")


def test_load_model_from_yaml():
    model = RuleClassifier(
        LesserThanNode("petal length (cm)", 1.9, 
                if_true=DummyRule(0), 
                if_false=CaseWhen([
                        LesserThan("petal length (cm)", 4.5, 1),
                        GreaterThan("petal length (cm)", 5.1, 2),
                        LesserThan("petal width (cm)", 1.4, 1),
                        GreaterThan("petal width (cm)", 1.8, 2),
                    ], default=1),
        ), 
        default=2
    )

    if os.path.exists("modeltest.yaml"):
        os.remove("modeltest.yaml")
    model.to_yaml("modeltest.yaml")
    assert os.path.exists("modeltest.yaml")
    model2 = RuleClassifier.from_yaml("modeltest.yaml")
    assert isinstance(model2, RuleClassifier)
    os.remove("modeltest.yaml")


def test_final_estimator():
    X, y = load_iris(return_X_y=True, as_frame=True)
    rules_plus_final_estimator = RuleClassifier(
        LesserThanNode("petal length (cm)", 1.9, 
                if_true=DummyRule(0), 
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

    rules_plus_final_estimator.fit(X, y)
    accuracy_score(y, rules_plus_final_estimator.predict(X)) > 0.98