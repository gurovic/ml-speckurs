# Formulation Progression: metrics and baseline

## Purpose

This map tracks how practice-task wording changes for the abstraction "quality metrics and baseline". The focus is on how much the task tells students about which metric to compute, where to compute it, what to compare it with, and how to use the result for a decision.

## Formulation Levels

| Level | Short name | What the wording gives the student |
|---|---|---|
| `F4` | Full recipe | Metric function/model names, variable names, comparison target, plotting/table requirements, and common mistakes. |
| `F3` | Guided implementation | Metric and comparison criterion are explicit; students choose part of the implementation. |
| `F2` | Protocol prompt | The task asks for a comparison/decision by metrics, but students assemble the evaluation table/workflow. |
| `F1` | Goal prompt | The task asks for an engineering conclusion; students infer which metric evidence is needed. |
| `F0` | Assumed habit | Metrics/baseline are not named directly but are expected as standard practice. |

## Current Trace

| Lesson | Task | Level | Evidence in wording | Interpretation |
|---:|---|---|---|---|
| 22 | Task 1. Release simulation | `F4` | Fixes `PRIMARY_METRIC = 'accuracy'` and asks students to explain why accuracy is a reasonable first metric. | First metric naming is explicit and lightweight. |
| 22 | Task 5. Baseline | `F4` | Names `DummyClassifier(strategy='most_frequent')`, `dummy_acc`, validation, and comparison with most frequent class share. | Baseline is introduced as a fully specified reference point. |
| 22 | Task 6. kNN vs baseline | `F4` | Names `KNeighborsClassifier(n_neighbors=5)`, `acc_raw`, bar chart, and condition `acc_raw > dummy_acc`. | Students practice direct metric comparison with no design burden. |
| 22 | Task 9. Confusion matrix and recall | `F4` | Names `confusion_matrix`, fixed class order, `recall_score(average=None)`, `recall_dragon`, and explanation target. | A new diagnostic metric is introduced with full scaffolding. |
| 24 | Task 6. Baseline set and kNN | `F4` | Gives baseline features, `Pipeline`, `KNeighborsRegressor(n_neighbors=5)`, validation MAE, and `mae_base`. | Regression baseline and MAE are explicit. |
| 24 | Task 7. Add feature groups | `F3` | Requires table "feature set -> MAE" and conclusion by validation numbers. | Comparison is guided, but students implement repeated evaluation. |
| 24 | Task 8. Combined feature set | `F3` | Requires final feature set based on previous MAE comparison and comparison with `mae_base`. | Students use metric evidence to select features. |
| 26 | Task 3. First baseline | `F3` | Requires baseline pipeline, validation MAE, additional metric or predictions, and saved result. | Baseline is familiar enough that exact model details are less prescribed. |
| 26 | Task 4. Improvement ideas | `F2` | Students try 2-4 ideas, compare by validation MAE, and check whether RMSE got worse. | Metrics become an experiment protocol. |
| 28 | Task 2. First baseline | `F3` | Requires probabilities, threshold 0.5, F1 and extra metrics on validation. | Classification metric workflow is guided. |
| 28 | Task 3. Threshold tuning | `F2` | Students sweep thresholds and maximize F1 on validation. | Metric drives decision-rule selection. |
| 30 | Task 1. Validation curve | `F4` | Explicitly requires train and validation MSE for polynomial degrees 1-15. | New diagnostic curve returns to full recipe. |
| 30 | Task 2. Overfitting diagnosis | `F3` | Choose `best_degree` by validation MSE and compare train/validation MSE. | Students interpret metrics rather than just compute them. |
| 32 | Task 1. Depth and overfitting | `F3` | Compare train and validation accuracy for several depths in a table. | Metric comparison is guided in a new model family. |
| 32 | Task 2. Tune `max_depth` | `F3` | Choose best depth by validation accuracy. | Short, familiar metric-driven selection. |
| 34 | Task 1. Three models | `F3` | Compare train/validation accuracy for tree, bagging, and random forest. | Students compare model families under one metric protocol. |
| 34 | Task 3. OOB score | `F3` | Compare OOB score with validation accuracy. | Introduces adjacent quality estimate with guided interpretation. |
| 36 | Task 1. Baseline and metrics | `F4` | Names `DummyRegressor(strategy='mean')`, MAE/RMSE/R2, validation, and `baseline_metrics`. | New capstone context restates metric baseline explicitly. |
| 36 | Task 6. Unified model ranking | `F2` | Collect all results, sort by validation RMSE, and plot model RMSE. | Students organize metric evidence across many models. |
| 36 | Task 8. Final engineering conclusion | `F1` | asks which model is best by validation, improvement over baseline, and final test choice. | Metrics become evidence for an engineering argument. |

## Pattern

- The course introduces each new metric family explicitly: accuracy/recall, MAE, F1, MSE curves, RMSE/R2.
- Baseline starts as a named model and variable, then becomes a required comparison habit.
- The end-state is strong: by lesson 36 students must use validation metrics, baseline improvement, speed, and support cost in one engineering conclusion.

## Additional Occurrences

The HTML visualization opens with the main anchor points of the progression. Its "all occurrences" mode also includes the wider scan below. These should be treated as supporting repetitions or adjacent uses.

| Lesson | Task | How the abstraction appears | Suggested status |
|---:|---|---|---|
| 22 | Task 2. Imports and split | Imports `accuracy_score`, `recall_score`, `confusion_matrix`; not yet used for reasoning in this task. | supporting setup |
| 22 | Task 7. Scaling and kNN | Computes `acc_scaled` on validation and compares with `acc_raw`. | supporting repetition |
| 22 | Task 10. Final test | Computes `test_acc` after choosing `best_k`. | final-check ritual |
| 24 | Tasks 2-5. Feature construction | Mentions train/validation because features must be built consistently; metric comparison comes later. | adjacent validation discipline |
| 24 | Task 9. Final summary | Asks which feature groups improved MAE and which did not. | supporting decision |
| 26 | Task 5. Final training and submission | Uses the model selected by validation results, but does not require new metric computation. | downstream use |
| 28 | Task 4. Improvement ideas | Compares `C`, `class_weight`, new features, and threshold by validation F1. | supporting decision |
| 28 | Task 5. Final training and submission | Applies the selected `best_threshold`; metric choice has already happened. | downstream use |
| 30 | Task 3. Learning curve | Uses `learning_curve`, 5-fold CV, and `neg_mean_squared_error`. | important supporting point |
| 30 | Task 4. K-fold CV | Uses `cross_validate`, validation MSE mean/std, and compares CV estimate with holdout validation. | important supporting point |
| 30 | Task 7. Selected model | Recomputes validation MSE for `best_degree`. | confirmation check |
| 30 | Task 8. Final summary | Requires conclusions about validation curve and learning curve based on results. | supporting synthesis |
| 32 | Task 4. `min_samples_leaf` | Compares validation accuracy for leaf-size constraints. | supporting model-selection repetition |
| 32 | Task 5. `ccp_alpha` | Uses validation accuracy to judge pruning. | supporting model-selection repetition |
| 32 | Task 6. Confusion matrix | Computes confusion matrix for the best tree on validation. | diagnostic metric repetition |
| 34 | Task 2. `n_estimators` | Plots number of trees against validation accuracy and identifies a plateau. | supporting model-selection repetition |
| 34 | Task 4. `max_depth` | Chooses best depth by validation accuracy, with tie-break toward simpler model. | supporting model-selection repetition |
| 34 | Task 6. Permutation importance | Measures quality drop on validation after feature permutation. | adjacent diagnostic use |
| 34 | Task 7. Train-validation gap | Explains why high train accuracy alone is not overfitting evidence. | diagnostic reasoning |
| 34 | Task 8. Final summary | Connects bagging/random forest conclusions to validation quality. | supporting synthesis |
| 36 | Task 2. Sklearn GBDT | Computes MAE/RMSE/R2 on validation and compares with baseline. | supporting model comparison |
| 36 | Task 3. GridSearchCV | Uses `neg_mean_absolute_error`, CV metric, and validation metrics for the best model. | important supporting point |
| 36 | Task 4. HistGradientBoosting | Compares validation metrics and training time with best GBDT. | supporting tradeoff |
| 36 | Task 5. External libraries | Computes MAE/RMSE/R2 on validation for XGBoost, LightGBM, and CatBoost. | supporting model comparison |
| 36 | Task 7. Library pros/cons | Requires quality/speed/support tradeoffs but not direct metric calculation. | adjacent engineering synthesis |

For a presentation, the anchor-point mode is easier to read. For curriculum repair, the all-occurrences mode is more useful because it reveals the repetition density.

## Candidate Progression Rule

For each repeated metric/baseline move:

| Encounter | Recommended wording |
|---:|---|
| 1 | Name the metric function/model, variable names, and comparison target. |
| 2 | Name the metric and target set; let students structure the table/plot. |
| 3 | Ask for comparison by metric and a decision. |
| 4 | Ask for an engineering conclusion backed by metrics. |
| 5+ | Grade absence of metric evidence as a reasoning error. |
