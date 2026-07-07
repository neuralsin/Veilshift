# Scientific Modules Theory

The core intelligence of Veilshift lies in the `science/` directory.

## 1. Feature Selection: QUBO (`science/qubo/module_g.py`)
Veilshift selects the optimal subset of features using **Quantum Unconstrained Binary Optimization (QUBO)**.
Instead of iteratively dropping features (which is prone to local minima), we cast feature selection as an energy minimization problem.

### The Formulation
We construct a matrix `Q` representing the quadratic interactions between features:
- **Diagonal Elements (Relevance):** We compute the absolute point-biserial correlation between each feature and the target label. This negative value encourages the selection of highly correlated features.
- **Off-Diagonal Elements (Redundancy):** We compute the absolute Pearson correlation between pairs of features. This positive penalty discourages selecting two features that provide the exact same information.

The optimal binary vector `x` (where x_i = 1 means feature i is selected) minimizes the energy function:  
`E(x) = x^T Q x`

## 2. Sensor Fusion (`science/fusion/module_h.py`)
Because sensors degrade under different conditions (e.g., Radar fails under heavy clutter, Thermal fails in high noise), Veilshift uses an adaptive fusion optimizer.

### Fusion Optimization
The pipeline trains independent Logistic Regression models for each sensor. We then fuse their probabilistic scores `s_i` using continuous weights `w_i`:  
`Score_fused = w_radar * s_radar + w_thermal * s_thermal + w_acoustic * s_acoustic`

The weights are strictly constrained:
- Sum to 1.0
- Must be positive (bounds between 0 and 1).

We optimize the weights using `scipy.optimize.minimize` with the SLSQP algorithm to maximize the ROC AUC of the fused scores.

### Threshold Selection (Neyman-Pearson)
Instead of an arbitrary 0.5 threshold, we use the Neyman-Pearson lemma. We fix a maximum allowable False Alarm Rate (FAR) - e.g., 5%. We sweep the ROC curve and select the absolute lowest threshold that strictly satisfies `FAR <= 0.05`. This maximizes the Detection Rate (Pd) without violating the operational constraints.
