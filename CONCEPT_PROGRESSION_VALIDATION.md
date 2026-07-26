# Concept Progression: validation

## Purpose

This map tracks how the concept of validation develops across lessons 1-36. It separates absence, first hints, introduction, practice, consolidation, extension, and transfer.

## Role Legend

| Role | Meaning |
|---|---|
| `absent` | The concept is not meaningfully used. |
| `hint` | A future need is mentioned, but students do not work with the concept yet. |
| `introduce` | The concept or tool is explicitly introduced. |
| `practice` | Students apply it directly in exercises. |
| `reinforce` | It is reused as part of another workflow. |
| `extend` | A more advanced version is introduced. |
| `transfer` | The concept is applied in a new model family or project format. |

## Trajectory

| Lessons | Role | What happens |
|---|---|---|
| 1-3 | `absent` | Data stack foundations. No model validation yet. |
| 4 | `hint` | First model appears; overfitting is mentioned as something the course will later address. |
| 5-20 | `absent` | Data analysis, visualization, statistics, clustering. These lessons build data handling habits, but not ML validation. |
| 21 | `introduce` | ML workflow introduces `train_test_split`, train / validation / test roles, baseline comparison, and the idea that preprocessing must be fit on train only. |
| 22 | `practice` | kNN practice uses train / validation / test, tunes `k` on validation, and keeps test as a final check. |
| 23 | `reinforce` | Feature engineering uses validation to check whether a new feature helps and reinforces `fit` on train, `transform` elsewhere. |
| 24 | `practice` | Feature engineering practice repeats split discipline and feature comparison on validation. |
| 25 | `reinforce` | Linear regression connects validation with metrics, overfitting risk, and model comparison. |
| 26 | `practice` | Regression practice applies validation in a leaderboard-like workflow. |
| 27 | `reinforce` | Logistic regression reuses validation with classification metrics, thresholds, and class imbalance. |
| 28 | `practice` | Classification practice repeats the validation protocol in a new task format. |
| 29 | `extend` | Dedicated theory lesson: overfitting, holdout validation, validation curve, learning curve, KFold, `cross_validate`, and `GridSearchCV`. |
| 30 | `practice` | Dedicated practice: students build validation and learning curves and use cross-validation. |
| 31 | `transfer` | Decision trees reuse validation for depth/complexity control and overfitting diagnosis. |
| 32 | `practice` | Tree practice checks train-validation gaps and model complexity. |
| 33 | `transfer` | Bagging and random forest connect validation with OOB, permutation importance, and ensemble overfitting. |
| 34 | `practice` | Ensemble practice compares train and validation quality and uses validation for importance/model comparison. |
| 35 | `transfer` | Boosting theory connects validation curves, learning rate, number of trees, and early stopping. |
| 36 | `practice` | Boosting practice applies train / validation / test, compares libraries, uses `GridSearchCV`, and reserves test for one final evaluation. |

## First Observations

- The full course axis should be 1-36, not only 21-36. Lessons 1-20 are important because they show prerequisites and the long pre-ML interval before validation becomes necessary.
- The first explicit validation protocol starts in lesson 21.
- The deepest conceptual expansion is concentrated in lessons 29-30.
- The transfer after lesson 30 is strong: validation reappears in trees, ensembles, and boosting.
- A possible next design step is to add a small pre-lesson-21 bridge task: "why evaluating a model on the same data can mislead us", using the first simple model from lesson 4.

## Evidence Scan

Automated keyword scan across notebooks and markdown found validation-related signals in lessons 4 and 21-36. The relevant patterns included `train_test_split`, `validation`, Russian stems for "validation" and "overfitting", `cross_validate`, `KFold`, `GridSearchCV`, `learning_curve`, `validation_curve`, `overfit`, and holdout-related wording.
