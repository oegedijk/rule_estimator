# rule_estimator

Inspired by the awesome [human learn](https://github.com/koaning/human-learn) package, this package makes it easy to build scikit-learn compatible business-rule estimators. These estimators can be stored to a human readable `.yaml` file, 
edited and then reloaded from such a `.yaml` file.

This estimator can be integrated into a scikit-learn `Pipeline`, including data
preprocessing steps. You can add a `final_estimator` for all cases where there is no applicable business rule, in which case they will be processed by this `final_estimator`.

There are two main usecases for this:
- When you have a sensitive application and you really want to have full
    transparancy and control over what your prediction model is doing, but you
    want to tie into the overall scikit-learn architecture instead of relying
    on long SQL scripts. 
- When you have some cases where you already know the correct label based on simple
    business rules, and it doesn't make sense to hope that the
    ML algorithm will correctly find this pattern and assign the same label.
    The remaining rows of data can be handled by an ML model. 

# Install

```sh
git clone https://github.com/oegedijk/rule_estimator.git
cd rule_estimator
pip install -e .
```

or 

```sh
pip install git+https://github.com/oegedijk/rule_estimator.git
```

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

Define the business rules. We start with a binary decision node, where all
flowers with a petal length below 1.9 get assigned label 0 (setosa).
For the remaining flowers we go through a CaseWhen list of a number of decision rules where if the condition holds either label 1 (versicolor) or label 2 (virginica) get applied. 
Any flowers not labeled get the default label=1 (versicolor):

```python
model = RuleClassifier(
    LesserThanNode("petal length (cm)", 1.9, # BinaryDecisionNode
        if_true=DummyRule(default=0), # DummyRule: always predict 0
        if_false=CaseWhen([
            # Go through these rules and if one applies, assign the prediction
            LesserThan("petal length (cm)", 4.5, prediction=1),
            GreaterThan("petal length (cm)", 5.1, prediction=2),
            LesserThan("petal width (cm)", 1.4, prediction=1),
            GreaterThan("petal width (cm)", 1.8, prediction=2),
        ], 
        default=1 # if no rule applies, assign prediction=1
        ),
    ), 
    default=2 # If no rule applied, assign prediction=2
)
```

Let's see how the rules performed:

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

You can also get a description of the business rule decision tree out:

```python
print(model.describe())
```

```
RulesClassifier
   BinaryDecisionNode petal length (cm) < 1.9
     DummyRule: Always predict 0
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

You can then store this model to a .yaml file. The description is added
as a summary comment on top:

```python
print(model.to_yaml())
```

```yaml
# RulesClassifier
#    BinaryDecisionNode petal length (cm) < 1.9
#      DummyRule: Always predict 0
#       Default: 0 
#      CaseWhen
#         If petal length (cm) < 4.5 then predict 1
#         If petal length (cm) > 5.1 then predict 2
#         If petal width (cm) < 1.4 then predict 1
#         If petal width (cm) > 1.8 then predict 2
#       Default: 1 
#  Default: 2 
__businessrule__:
  module: rule_estimator.core
  name: RuleClassifier
  description: RulesClassifier
  params:
    rules:
      __businessrule__:
        module: rule_estimator.business_rules
        name: LesserThanNode
        description: BinaryDecisionNode petal length (cm) < 1.9
        params:
          col: petal length (cm)
          cutoff: 1.9
          if_true:
            __businessrule__:
              module: rule_estimator.business_rules
              name: DummyRule
              description: 'DummyRule: Always predict 0'
              params:
                default: 0
          if_false:
            __businessrule__:
              module: rule_estimator.business_rules
              name: CaseWhen
              description: CaseWhen
              params:
                rules:
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: LesserThan
                    description: If petal length (cm) < 4.5 then predict 1
                    params:
                      col: petal length (cm)
                      cutoff: 4.5
                      prediction: 1
                      default: null
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: GreaterThan
                    description: If petal length (cm) > 5.1 then predict 2
                    params:
                      col: petal length (cm)
                      cutoff: 5.1
                      prediction: 2
                      default: null
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: LesserThan
                    description: If petal width (cm) < 1.4 then predict 1
                    params:
                      col: petal width (cm)
                      cutoff: 1.4
                      prediction: 1
                      default: null
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: GreaterThan
                    description: If petal width (cm) > 1.8 then predict 2
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

If it looks good, store it to a file:

```python
model.to_yaml("iris_rules.yaml")
```

You can then go and edit this .yaml (e.g. adjust a cutoff) and reload
the model with the new cutoff from the .yaml:

```python
loaded_model = RuleClassifier.from_yaml("iris_rules.yaml")
loaded_model.predict(X)
```

## Including a `final_estimator`

You can also add a final estimator, which can be any scikit-learn compatible estimator (such as `DecisionTreeClassifier`, `RandomForestClassifier`, etc).

Any cases not covered by a DecisionRule will result in a `np.nan` in the 
predictions array `y`. These will then be estimated by the `final_estimator`.

By default the `final_estimator` gets fitted on the remaining cases only (i.e. those not handled by a DecisionRule), but you can pass `fit_remaining_only=False`
to fit on the entire dataset `X` instead:

```python
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

```

This seems to improve performance (training on the test set FTW!):
```python
print(classification_report(y, rules_plus_final_estimator.predict(X)))
```
```
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        50
           1       0.96      1.00      0.98        50
           2       1.00      1.00      1.00        50

    accuracy                           0.99       150
   macro avg       0.99      0.99      0.99       150
weighted avg       0.99      0.99      0.99       150
```

# Defined BusinessRules

Currently the following BusinessRules are defined in the library:


-  `DummyRule`: simply always assign a `default` label
-  `GreaterThan`: if `col` is greater than `cutoff` assign `prediction`
-  `GreaterEqualThan`: if `col` is greater or equal than `cutoff` assign `prediction`
-  `LesserThan`: if `col` if lesser then `cutoff` assign `prediction`
-  `LesserEqualThan`: if `col` if lesser or equal than `cutoff` assign `prediction`
-  `CaseWhen`: process a list of `BusinessRules` one-by-one, if a rule applies
    assign the prediction, then pass the remaining rows to the next Rule, etc.
    
There are also four `BinaryDecisionNodes` defined. These evaluate a condition,
and if the condition holds pass the prediction off to `BusinessRule` `if_true`,
and otherwise to `BusinessRule` `if_false`:

-  `GreaterThanNode`
-  `GreaterEqualThanNode`
-  `LesserThanNode`
-  `LesserEqualThanNode`
# Defining your own BusinessRules

It is easy to define and add your own BusinessRules, the basic structure is:

```python
class GreaterThan(BusinessRule):
    def __init__(self, col:str, cutoff:float, prediction:Union[float, int], default=None):
        super().__init__()

    def predict(self, X:pd.DataFrame):
        y = np.where(X[self.col] > self.cutoff, self.prediction, self.default)
        return y

    def __rulerepr__(self):
        return f"If {self.col} > {self.cutoff} then predict {self.prediction}"
```

The `super().__init__()` automatically assigns all `__init__` parameters to
attributes (so you don't have to add boilerplate like `self.cutoff=cutoff`), and also
adds them to a `._stored_params` dict that can later be exported to `.yaml`. It
also automatically converts parameter `default` to `np.nan` if it is `None` (and defines it for you if you didn't have it in the parameters).

The `predict(X)` method should return a `np.array` `y` with predictions for
all cases where the rule applies, and `np.nan` otherwise (or `self.default`).

The dundermethod `__rulerepr__` returns a human readable interpretation of your
rule. This gets displayed when you call `rule.describe()`, and gets added to
the `.yaml` as well. It defaults to `'BusinessRule'` which is not very descriptive, so worth the effort of replacing it with something better. 