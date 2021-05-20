# rule_estimator

This package makes it easy to build scikit-learn compatible business-rule estimators.


# Example

## Load imports and data

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from rule_estimator import *

X, y = load_iris(return_X_y=True, as_frame=True)

```

## Instantiate RuleClassifier

```python
model = RuleClassifier(
    LesserThanNode("petal length (cm)", 1.9, 
        if_true=DefaultRule(0), 
        if_false=CaseWhen([
            LesserThan("petal length (cm)", 4.5, 1),
            GreaterThan("petal length (cm)", 5.1, 2),
            LesserThan("petal width (cm)", 1.4, 1),
            GreaterThan("petal width (cm)", 1.8, 2),
        ], default=1),
    ), 
    default=2
)
```

```python
print(classification_report(y, model.predict(X)))
```
```
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        50
           1       0.83      1.00      0.91        50
           2       1.00      0.84      0.91        50

    accuracy                           0.93       150
   macro avg       0.94      0.93      0.93       150
weighted avg       0.94      0.93      0.93       150
```

```python
print(model.describe())
```

```
RulesClassifier
   BinaryDecisionNode petal length (cm) < 1.9
     Always predict 0
      Default: 0 
     CaseWhen
        If petal length (cm) < 4.5 then predict 1
        If petal length (cm) > 5.1 then predict 2
        If petal width (cm) < 1.4 then predict 1
        If petal width (cm) > 1.8 then predict 2
      Default: 1 
 Default: 2 
```

## Storing model to `.yaml`

```python
print(model.to_yaml())
```

```yaml


# RulesClassifier
#    BinaryDecisionNode petal length (cm) < 1.9
#      Always predict 0
#       Default: 0 
#      CaseWhen
#         If petal length (cm) < 4.5 then predict 1
#         If petal length (cm) > 5.1 then predict 2
#         If petal width (cm) < 1.4 then predict 1
#         If petal width (cm) > 1.8 then predict 2
#       Default: 1 
#  Default: 2 
__storable__:
  module: __main__
  name: RuleClassifier
  params:
    rules:
      __storable__:
        module: __main__
        name: LesserThanNode
        params:
          col: petal length (cm)
          cutoff: 1.9
          businessrule_true:
            __storable__:
              module: __main__
              name: DefaultRule
              params:
                default: 0
          businessrule_false:
            __storable__:
              module: __main__
              name: CaseWhen
              params:
                rules:
                - __storable__:
                    module: __main__
                    name: LesserThan
                    params:
                      col: petal length (cm)
                      cutoff: 4.5
                      prediction: 1
                      default: null
                - __storable__:
                    module: __main__
                    name: GreaterThan
                    params:
                      col: petal length (cm)
                      cutoff: 5.1
                      prediction: 2
                      default: null
                - __storable__:
                    module: __main__
                    name: LesserThan
                    params:
                      col: petal width (cm)
                      cutoff: 1.4
                      prediction: 1
                      default: null
                - __storable__:
                    module: __main__
                    name: GreaterThan
                    params:
                      col: petal width (cm)
                      cutoff: 1.8
                      prediction: 2
                      default: null
                default: 1
    default: 2
    final_estimator: null
    fit_remaining_only: true
```

Store the estimator configuration to file and recover from yaml:

```python
model.to_yaml("iris_rules.yaml")
loaded_model = RuleClassifier.from_yaml("iris_rules.yaml")
loaded_model.predict(X)
```

## Including a `final_estimator`

```python
model2 = RuleClassifier(
    LesserThanNode("petal length (cm)", 1.9, 
               if_true=DefaultRule(0), 
               if_false=CaseWhen([
                    LesserThan("petal length (cm)", 4.5, 1),
                    GreaterThan("petal length (cm)", 5.1, 2),
                    LesserThan("petal width (cm)", 1.4, 1),
                    GreaterThan("petal width (cm)", 1.8, 2),
                ]),
    ), 
    final_estimator=DecisionTreeClassifier()
)

model2.fit(X, y)
print(classification_report(y, model2.predict(X)))

print(model2.describe())
```