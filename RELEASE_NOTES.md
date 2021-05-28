

## Version 0.2.1:
### Breaking Changes
- 
- 

### New Features
- new `estimator.parallel_coordinates(X, y, rule_id)` method to plot a parallel 
    coordinates plot of data entering rule `rule_id`.
- new rules: `MultiRangeAndRule` and `MultiRangeOrRule`.
- new nodes: `MultiRangeAndNode` and `MultiRangeOrNode`.

### Bug Fixes
- Fixes bugs with `replace_rule` and `append_rule`
-

### Improvements
- `append_rule` now also inserts in the correct position when `rule_id` is
    inside a `CaseWhen` rule
-

### Other Changes
-
-


## version 0.2:
### Breaking Changes
- Custom rules are now defined with `__rule__` method that returns a boolean mask
    instead of with `predict(X)` method.
- `DummyRule` is now called `PredictionRule`


### New Features
- each rule now gets assigned a `rule_id`, which is displayed when you call
    `estimator.describe()`
- new `score_rules(X, y)` method that shows performance of individual rules
- new `get_igraph()` method, that returns an igraph Graph object of the rules
- new `plot()` method that returns a plotly figure of the rules
- new `get_rule(rule_id)`, `replace_rule(rule_id, new_rule)` and `append_rule(rule_id, new_rule)` methods
- new `get_rule_params(rule_id)` and `set_rule_params(rule_id, **params)` methods
- new `get_rule_input(rule_id, X, y)` and `get_rule_leftover(rule_id, X, y)` to get the specific data
    that either flows into a rule, or the unlabeled data that flows out of a rule.
    This helps in constructing new rules as you can target it to the data
    that would appear in that part of the rule graph. 


### Improvements
- data is now split up and only non-labeled data is passed to downstream rules.
-



## Template:
### Breaking Changes
- 
- 

### New Features
-
-

### Bug Fixes
-
-

### Improvements
-
-

### Other Changes
-
-