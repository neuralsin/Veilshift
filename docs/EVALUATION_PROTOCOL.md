# Strict Out-Of-Fold (OOF) Evaluation Protocol

Veilshift mathematically prevents data leakage (overfitting) by isolating all evaluation metrics to an Out-Of-Fold protocol. This is implemented in `science/evaluation/oof_protocol.py`.

## The Problem with Traditional Machine Learning
If you select features, optimize fusion weights, or calculate a threshold on the exact same data you use to evaluate performance, the model "memorizes" the test set. It looks like it performs flawlessly, but will fail catastrophically in production.

## The Veilshift OOF Solution
We execute a strict nested 5-Fold Stratified K-Fold protocol.

For each of the 5 outer folds, the data is split into:
- **Outer Train (80%)**
- **Outer Test (20%)**

### 1. The Inner Loop (Training)
We take the **Outer Train** subset and further split it:
- **Inner Train (80% of Outer Train):** We calculate the QUBO matrix, select the features, train the independent sensor Logistic Regression models, and optimize the Fusion weights here.
- **Inner Val (20% of Outer Train):** We generate predictions on this tiny validation set purely to select the strict Neyman-Pearson threshold that guarantees FAR <= 5%.

### 2. The Refit
We take the exact features selected from the Inner Train, the exact weights optimized from the Inner Train, and the exact threshold selected from the Inner Val. We then refit the Logistic Regression classifiers on the *entire* Outer Train set.

### 3. The Strict Test
We generate the final scores on the **Outer Test** set. 
Because the Outer Test set was completely hidden during feature selection, weight optimization, and thresholding, these predictions are mathematically "clean."

### 4. Pooling
We repeat this 5 times until every single sample in the entire dataset has been predicted exactly once as an Outer Test sample. We pool all these predictions together and calculate our final canonical metrics (AUC, Pd, FAR).

This guarantees the numbers shown in Mission Control are uncheated, robust, and mathematically sound.
