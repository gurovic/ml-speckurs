# Formulation Progression: validation tasks

## Purpose

This map tracks how practice-task wording changes as students repeatedly meet the validation concept. It focuses on the amount of scaffolding in the task statement, not on the mathematical difficulty of the topic.

## Formulation Levels

| Level | Short name | What the wording gives the student | Expected student autonomy |
|---|---|---|---|
| `F4` | Full recipe | Function/tool names, constants, variable names, order of steps, checks, and common mistakes. | Follow a precise recipe. |
| `F3` | Guided implementation | The target action and important constraints are explicit; function names or constants may be partly omitted. | Choose some syntax/details. |
| `F2` | Protocol prompt | The task names the protocol and quality criterion, but students must design a reasonable implementation. | Assemble a familiar workflow. |
| `F1` | Goal prompt | The desired outcome is stated; students infer the validation action/tool from context. | Decide which validation move is needed. |
| `F0` | Assumed habit | Validation is not requested directly; the student is expected to apply it as standard practice. | Notice and apply the habit independently. |

## Current Validation Task Trace

| Lesson | Task | Level | Evidence in wording | Interpretation |
|---:|---|---|---|---|
| 22 | Task 2. Imports and split | `F4` | Names `train_test_split`, all imports, `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`, `VAL_SIZE = 0.25`, two split steps, expected variable names, and size checks. | First encounter is intentionally recipe-like. Good for introducing the mechanics. |
| 22 | Task 5. Baseline | `F4` | Names `DummyClassifier(strategy='most_frequent')`, train fit, validation metric, and comparison target. | Still explicit: the student practices "train on train, measure on validation" with no tool-choice burden. |
| 22 | Task 8. Tune `k` | `F4` | Gives exact `k` list, output table names, metric name, and graph requirement. | Hyperparameter selection is introduced as a fully specified loop. |
| 22 | Task 10. Final test | `F4` | Explicitly says test is used after `best_k`; warns against test-based tuning. | Good first final-test ritual. |
| 24 | Task 1. Split | `F3` | Gives train/validation tables, 70/30, `RANDOM_STATE`, and fit-on-train rule, but does not spell out every split variable. | Slightly less recipe-like: students implement a known split in a new data shape. |
| 24 | Tasks 4-5. Encoders/imputation | `F3` | Names encoders/imputation logic and repeatedly says fit on `flats_train`, apply to validation. | Validation discipline moves into preprocessing, not just model scoring. |
| 26 | Task 2. Validation and preprocessing | `F2` | Says split `train.csv` into train/validation with fixed random state and avoid leakage via `ColumnTransformer` or pipeline. | Students now assemble the workflow in a competition format. |
| 26 | Task 4. Improvement ideas | `F2` | Gives example ideas but requires comparing them only on validation and preserving the same validation sample. | The concept becomes an experiment protocol. |
| 28 | Task 1. Validation and stratification | `F2` | Requires stratified train/validation split, class distribution checks, and no test use. | Less mechanical than lesson 22; adds class-distribution reasoning. |
| 28 | Task 3. Threshold tuning | `F2` | Requires threshold sweep on validation and saving `best_threshold`, but students organize the sweep. | Validation controls a decision rule, not only model choice. |
| 30 | Task 0. Imports and split | `F4` | Again names all tools, `RANDOM_STATE = 42`, split ratio 70/30, and variables. | Because the topic is new/deep, scaffolding returns to full recipe. This is pedagogically reasonable. |
| 30 | Task 1. Validation curve | `F4` | Gives degrees 1-15, fit/transform protocol, train/validation MSE, and chart. | Advanced concept is introduced with full operational detail. |
| 30 | Task 2. Overfitting diagnosis | `F3` | Requires choosing `best_degree` by validation MSE and explaining train-validation gap. | Still guided, but now the student must interpret the result. |
| 30 | Task 4. K-fold CV | `F3` | Names `cross_validate`, 5-fold, mean/std, and the `test_score` caveat. | High support remains because CV terminology is easy to misread. |
| 32 | Task 0. Split and imports | `F4` | Names imports, `RANDOM_STATE=42`, `test_size=0.35`, `stratify=y`, and notes that sklearn's `test_size` is used as validation. | New model family gets a fully specified validation setup. |
| 32 | Task 2. Tune `max_depth` | `F3` | Gives range 1-12 and requires choosing by validation accuracy. | The loop is specified, but the code structure is less step-by-step. |
| 32 | Tasks 4-5. Leaf/pruning constraints | `F3` | Gives parameter lists and requires validation accuracy and conclusion. | Students practice model-complexity control through validation. |
| 34 | Task 0. Split | `F4` | Names imports, 70/30 split, `stratify=y`, and `RANDOM_STATE=42`. | Explicit reset for a new ensemble topic. |
| 34 | Task 2. `n_estimators` | `F3` | Gives exact list, metric, graph, and plateau interpretation. | Guided experiment; students interpret diminishing returns. |
| 34 | Task 3. OOB score | `F3` | Names `oob_score=True`, validation comparison, and interpretation of disagreement. | Introduces an adjacent validation-like estimate with guided comparison. |
| 34 | Task 7. Train-validation gap | `F2` | Asks for a markdown explanation and gives a conceptual hint. | Moves from coding recipe to diagnostic reasoning. |
| 36 | Task 0. Loading and split | `F3` | Requires train/validation/test and final-test discipline, but no exact split constants are specified in the wording. | Students are expected to reproduce the standard split protocol. |
| 36 | Task 3. `GridSearchCV` | `F3` | Names `GridSearchCV`, required parameters, scoring, CV metric, and validation check. | Advanced tool is named, but students design the parameter grid. |
| 36 | Task 6. Unified ranking | `F2` | Requires collecting model results, sorting by validation RMSE, and charting. | Validation becomes the organizing criterion for comparison. |
| 36 | Task 8. Final engineering conclusion | `F1` | Asks which model is best by validation and then one final test evaluation. | The validation protocol is expected as part of an engineering decision. |

## Pattern

The current course does not simply fade support linearly. It follows a better pattern:

| Segment | Wording pattern |
|---|---|
| 22 | Full recipe for first mechanics. |
| 24-28 | Gradual release into reusable workflow and competition-style decisions. |
| 30 | Scaffolding intentionally returns because validation curve, learning curve, and CV are new conceptual tools. |
| 32-34 | Support fades again while transferring validation to trees and ensembles. |
| 36 | Students mostly use validation as an engineering selection protocol. |

## Design Implications

- Keep `F4` when a new validation tool appears for the first time: `train_test_split`, validation curve, learning curve, `cross_validate`, `GridSearchCV`, OOB.
- After two appearances of the same move, lower the formulation level by one step: from exact constants and variables to "use the standard split protocol".
- Add explicit "do not use test for selection" reminders longer than other details. This is a safety habit, not just syntax.
- Track concept and formulation separately. A lesson can be conceptually advanced but formulation-explicit, as lesson 30 is.
- The next audit should detect gaps where an advanced task jumps from `F4` to `F1` without an intermediate `F2/F3` practice.

## Candidate Progression Rule

For each repeated validation move:

| Encounter | Recommended wording |
|---:|---|
| 1 | Give function, constants, variable names, and checks. |
| 2 | Give function and constraints; let students choose variable names. |
| 3 | State the protocol and target metric; let students assemble code. |
| 4 | State the modeling goal; students infer validation is needed. |
| 5+ | Treat validation as a required habit and grade misuse of test/leakage. |
