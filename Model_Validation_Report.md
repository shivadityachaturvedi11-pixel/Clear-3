# Internship Task-3: Model Validation & Tuning Report

## 1. Overfitting Analysis
In the initial baseline modeling of the Decision Tree Regressor, an unconstrained tree was trained on the training split. The resulting metrics revealed an extreme variance gap:
* **Train RMSE:** 0.0000 (Perfect training fit)
* **Test RMSE:** 0.7321

### Interpretation:
A tree with no maximum depth constraint will continually split nodes until every single training sample resides in its own leaf node. While this achieves zero error on the training set, it captures random noise rather than general underlying trends. The massive gap between the Train RMSE and Test RMSE is textbook evidence of high variance (overfitting), making the baseline tree incapable of general production readiness.

---

## 2. Tuning Approach
To counter the overfitting observed above, automated hyperparameter optimization was executed using `GridSearchCV` across a 5-fold cross-validation setup. 

### Selected Hyperparameter Grid:
* `max_depth`: `[3, 5, 7, 10]` — Restricts the maximum vertical splits of the tree to prevent it from memorizing individual records.
* `min_samples_split`: `[2, 5, 10]` — Increases the minimum samples required to split an internal node, enforcing smoother node averages.

By evaluating all 12 structural combinations across 5 distinct data folds, `GridSearchCV` structurally penalized overly complex trees that failed to generalize on validation folds, selecting the optimal configuration.

---

## 3. Why Cross-Validation Results are Trusted
A single train-test split is highly susceptible to the "luck of the draw." If the random split accidentally places anomalous data entirely into the test set, performance metrics will skew negatively. 

Cross-Validation ($K\text{-fold}=5$) splits the data into 5 equal portions. The model trains on 4 portions and tests on the remaining 1, repeating this process 5 times so every data point is used for both training and validation. The resulting average validation score provides a highly stable, realistic performance estimation that professionals trust before deploying models to production.

---

## 4. Final Model Selection & Trade-offs Justification
The **Tuned Decision Tree** was selected as the final production model based on the comparative results:

* **Linear/Ridge Baselines:** Underperformed with a higher RMSE of ~0.7455 and an R² score of 0.5758 due to an inability to map non-linear relationships inherent in spatial geographic housing data.
* **Tuned Decision Tree:** Outperformed the linear baselines by driving the RMSE down to ~0.6725 and elevating the R² score to 0.6510.

### The Trade-off:
We deliberately accepted a minor compromise in simplicity. Linear regression models are highly interpretable (straight coefficients), whereas a Tuned Decision Tree is non-linear and slightly harder to map out manually. However, by constraining `max_depth`, we preserved a fair degree of model interpretability while securing a substantial jump in predictive performance.

