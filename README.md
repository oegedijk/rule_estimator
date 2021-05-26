# rule_estimator

Inspired by the awesome [human learn](https://github.com/koaning/human-learn) package, 
this package makes it easy to build scikit-learn compatible business-rule estimators. 
These estimators can be stored to a human readable `.yaml` file, 
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
pip install rule-estimator
```

# Example

## Load imports and data

```python
from sklearn.datasets import load_iris
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
    LesserThanNode("petal length (cm)", 1.91, # BinaryDecisionNode
        if_true=PredictionRule(prediction=0), # DummyRule: always predict 0
        if_false=CaseWhen([
                    # Go through these rules and if one applies, assign the prediction
                    LesserThan("petal length (cm)", 4.5, prediction=1),
                    GreaterThan("petal length (cm)", 5.1, prediction=2),
                    LesserThan("petal width (cm)", 1.4, prediction=1),
                    GreaterThan("petal width (cm)", 1.8, prediction=2),
                ], default=1 # if no rule applies, assign prediction=1
            ),
    ), 
```

Let's see how the rules performed:

```python
from sklearn.metrics import classification_report
print(classification_report(y, model.predict(X)))
```
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        50
           1       0.86      1.00      0.93        50
           2       1.00      0.84      0.91        50

    accuracy                           0.95       150
   macro avg       0.95      0.95      0.95       150
weighted avg       0.95      0.95      0.95       150
```

Pretty good! You can also get a description of the business rule decision tree out:

```python
print(model.describe())
```

```
RuleClassifier
  0: LesserThanNode: petal length (cm) < 1.91
    1: PredictionRule: Always predict 0
    2: CaseWhen (default=1)
       3: LesserThan: If petal length (cm) < 4.5 then predict 1
       4: GreaterThan: If petal length (cm) > 5.1 then predict 2
       5: LesserThan: If petal width (cm) < 1.4 then predict 1
       6: GreaterThan: If petal width (cm) > 1.8 then predict 2
```

If you have `plotly` installed, you can also call `model.plot()` to get
a graphic depiction of the decision tree.

## Storing model to `.yaml`

You can then store this model to a .yaml file. The description is added
as a summary comment on top. Storing the model inside a configuration file
make it transparant what the model does exactly, allows anyone to adjust
the working of the model with a simple text editor, plus you can check it
into version control!

```python
print(model.to_yaml())
```

```yaml
# RuleClassifier
#   0: LesserThanNode: petal length (cm) < 1.91
#     1: PredictionRule: Always predict 0
#     2: CaseWhen (default=1)
#        3: LesserThan: If petal length (cm) < 4.5 then predict 1
#        4: GreaterThan: If petal length (cm) > 5.1 then predict 2
#        5: LesserThan: If petal width (cm) < 1.4 then predict 1
#        6: GreaterThan: If petal width (cm) > 1.8 then predict 2
__businessrule__:
  module: rule_estimator.core
  name: RuleClassifier
  description: RuleClassifier
  params:
    rules:
      __businessrule__:
        module: rule_estimator.business_rules
        name: LesserThanNode
        description: 'LesserThanNode: petal length (cm) < 1.91'
        params:
          col: petal length (cm)
          cutoff: 1.91
          if_true:
            __businessrule__:
              module: rule_estimator.business_rules
              name: PredictionRule
              description: 'PredictionRule: Always predict 0'
              params:
                prediction: 0
          if_false:
            __businessrule__:
              module: rule_estimator.core
              name: CaseWhen
              description: CaseWhen
              params:
                rules:
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: LesserThan
                    description: 'LesserThan: If petal length (cm) < 4.5 then predict
                      1'
                    params:
                      col: petal length (cm)
                      cutoff: 4.5
                      prediction: 1
                      default: null
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: GreaterThan
                    description: 'GreaterThan: If petal length (cm) > 5.1 then predict
                      2'
                    params:
                      col: petal length (cm)
                      cutoff: 5.1
                      prediction: 2
                      default: null
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: LesserThan
                    description: 'LesserThan: If petal width (cm) < 1.4 then predict
                      1'
                    params:
                      col: petal width (cm)
                      cutoff: 1.4
                      prediction: 1
                      default: null
                - __businessrule__:
                    module: rule_estimator.business_rules
                    name: GreaterThan
                    description: 'GreaterThan: If petal width (cm) > 1.8 then predict
                      2'
                    params:
                      col: petal width (cm)
                      cutoff: 1.8
                      prediction: 2
                      default: null
                default: 1
          default: null
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

If you would like the original code definition back, simply call `print(loaded_model.to_code())`.

## Scoring the rules

You can check the performance of each rule. You see how many inputs went into a certain rule (`n_inputs`), and to how many of those inputs the rules applied (`n_outputs`). Coverage is then the ratio of the two.

For RuleClassifier the accuracy and for RuleRegressor the root mean squared error is computed per rule. Default predictions are scored seperately.

```python
model.score_rules(X, y)
```
|    |   rule_id | name           | description                                            |   prediction |   n_inputs |   n_outputs |   coverage |   accuracy |
|---:|----------:|:---------------|:-------------------------------------------------------|-------------:|-----------:|------------:|-----------:|-----------:|
|  0 |         0 | LesserThanNode | LesserThanNode: petal length (cm) < 1.91               |          nan |        150 |         150 |  1         |   0.946667 |
|  1 |         1 | PredictionRule | PredictionRule: Always predict 0                       |            0 |         50 |          50 |  1         |   1        |
|  2 |         2 | CaseWhen       | CaseWhen                                               |          nan |        100 |          74 |  0.74      |   1        |
|  3 |         2 | ↳              | default: predict 1                                     |            1 |        100 |          26 |  0.26      |   0.692308 |
|  4 |         3 | LesserThan     | LesserThan: If petal length (cm) < 4.5 then predict 1  |            1 |        100 |          29 |  0.29      |   1        |
|  5 |         4 | GreaterThan    | GreaterThan: If petal length (cm) > 5.1 then predict 2 |            2 |         71 |          34 |  0.478873  |   1        |
|  6 |         5 | LesserThan     | LesserThan: If petal width (cm) < 1.4 then predict 1   |            1 |         37 |           3 |  0.0810811 |   1        |
|  7 |         6 | GreaterThan    | GreaterThan: If petal width (cm) > 1.8 then predict 2  |            2 |         34 |           8 |  0.235294  |   1        |

You can see that the main error of the model is due to the default prediction (default=1) for the CaseWhen rule with rule_id 2.

## Getting rule input rows

In order to improve a rule it is useful to get a snapshot of the data flowing into that rule. You can get that with `get_rule_input()`.
This allows you to investigate whether you could come up with a better rule with that same data. You can also get the leftover data, that is the data that flows into a particular rule, but does not get labeled. If you pass both `X` and `y` you get both back. If you only pass `X`, you only get `X` back. So to retrieve the data flowing into the rule with rule_id 4:

```python
input_X, input_y = model.get_rule_input(4, X, y)
leftover_X = model.get_rule_leftover(4, X)
```

## Rule suggestions

You can also ask the model to compute a rule seggestion at the location of a particular rule based on a DecisionTree with depth 1.
You can either get a `kind='rule'`, `'prediction'` or '`node`' suggestion. You can also get a suggestion based
on the leftover data of a rule (`after=True`).

```python
model.suggest_rule(6, X, y)
model.suggest_rule(6, X, y, kind='node', after=True)
```

## Retrieving and setting rule parameters

You can retrieve and update rule parameters based on their rule_id:

```python
params = model.get_rule_params(3)
model.set_rule_params(3, cutoff=4.6)
```

## Retrieve, replace or append rule

You can retrieve a rule by its rule_id:

```python
rule = model.get_rule(5)
```

And then you can replace the entire rule. Or you can append a rule to it. 
This will turn the rule into a `CaseWhen` rule with at least two components (the 
original rule and the appended rule). If the rule is already a `CaseWhen` then the rule 
simply gets appended to the end.

```
model.replace_rule(5, LesserThan(col='petal width (cm)', cutoff=1.5, prediction=1))
model.append_rule(5, GreaterThan(col='petal width (cm)', cutoff=4.5, prediction=2))
```

## Including a `final_estimator`

You can also add a final estimator, which can be any scikit-learn compatible estimator (such as `DecisionTreeClassifier`, `RandomForestClassifier`, etc).

Any cases not covered by a DecisionRule will result in a `np.nan` in the 
predictions array `y`. These will then be estimated by the `final_estimator`.

By default the `final_estimator` gets fitted on the remaining cases only (i.e. those not handled by a DecisionRule), but you can pass `fit_remaining_only=False`
to fit on the entire dataset `X` instead:

```python
from sklearn.tree import DecisionTreeClassifier

rules_plus_final_estimator = RuleClassifier(
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


-  `EmptyRule`: always predict `np.nan`.
-  `PredictionRule`: simply always assign `prediction` label
-  `GreaterThan`: if `col` is greater than `cutoff` assign `prediction`
-  `GreaterEqualThan`: if `col` is greater or equal than `cutoff` assign `prediction`
-  `LesserThan`: if `col` if lesser then `cutoff` assign `prediction`
-  `LesserEqualThan`: if `col` if lesser or equal than `cutoff` assign `prediction`

If you do not pass a `default` parameter to these rules, then any rows not covered
will get a `np.nan` prediction. 

`CaseWhen` processes a list of `BusinessRules` one-by-one, if a rule applies
    it assigns the prediction, then passes the remaining rows to the next Rule, etc.
    
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
class VersicolorRule(BusinessRule):
    def __init__(self, length_cutoff=4.6, width_cutoff=1.5, prediction=1, default=2):
        super().__init__()

    def __rule__(self, X):
        return (X['petal length (cm)'] < self.length_cutoff) | (X['petal width (cm)'] < self.width_cutoff)

    def __rulerepr__(self):
        return f"VersicolorRule: if petal length < {self.length_cutoff} or petal width < {self.width_cutoff} predict 1"
```

The `super().__init__()` automatically assigns all `__init__` parameters to
attributes (so you don't have to add boilerplate like `self.length_cutoff=length_cutoff`), and also
adds them to a `._stored_params` dict that can later be exported to `.yaml`. It
also automatically adds attributes `prediction` and `default`, even when they are not 
defined in the init, and converts them to `np.nan` if they are `None`. 

The dundermethod `__rule__` then defines the actual rule and takes in a `pd.DataFrame` `X`
of input data. It should return a boolean `pd.Series` with `True` for the rows where
the rule applies and `False` where it does not. Where the `__rule__` is `True` it will assign
`prediction`, where it is `False` it will either assign `np.nan` or `default`.


The dundermethod `__rulerepr__` returns a human readable interpretation of your
rule. This gets displayed when you call `rule.describe()`, and gets added to
the `.yaml` as well. It defaults to `'BusinessRule'` which is not very descriptive, 
so worth the effort of replacing it with something better. 


```python
model = RuleClassifier(
    LesserThanNode("petal length (cm)", 1.91, # BinaryDecisionNode
        if_true=PredictionRule(prediction=0), # PredictionRule: always predict 0
        if_false=VersicolorRule()
    ),   
)

model.describe()
```

Here's the output:
```
RuleClassifier
  0: LesserThanNode: petal length (cm) < 1.91
    1: PredictionRule: Always predict 0
    2: VersicolorRule: if petal length < 4.6 or petal width < 1.5 predict 1 (default=2)
```