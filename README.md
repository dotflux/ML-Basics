# Machine Learning Fundamentals & Practical Workflows

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Environment-Jupyter%20Notebook-F37626.svg)](https://jupyter.org/)

A structured collection of practical machine learning implementations covering end-to-end data workflows: exploratory data analysis, statistical feature selection, regression, multi-model classification benchmarking, hyperparameter optimization, ensemble architectures, dimensionality reduction, and clustering algorithms.

---

## Repository Structure

```text
.
|-- csv/
|   `-- insurance.csv
|-- ensemble.ipynb
|-- gridAndRandomSearchCV.ipynb
|-- insurance.ipynb
|-- pca.ipynb
|-- titanic.ipynb
`-- unsupervised.ipynb
```

---

## Overview of Notebooks

| Notebook                      | Core Machine Learning Concept                | Primary Algorithms                                            | Dataset                        | Evaluation Metrics                                |
| :---------------------------- | :------------------------------------------- | :------------------------------------------------------------ | :----------------------------- | :------------------------------------------------ |
| `insurance.ipynb`             | Supervised Regression & Feature Selection    | Multiple Linear Regression                                    | `csv/insurance.csv`            | R2 Score, Adjusted R2                             |
| `titanic.ipynb`               | Supervised Classification & Cross-Validation | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVC     | Seaborn Titanic                | Accuracy, Confusion Matrix, Precision, Recall, F1 |
| `gridAndRandomSearchCV.ipynb` | Hyperparameter Optimization                  | KNN, Support Vector Classifier (SVC)                          | Iris Dataset                   | Accuracy, Best Cross-Validated Score              |
| `ensemble.ipynb`              | Ensemble Architectures                       | Stacking, Random Forest, AdaBoost, Gradient Boosting, XGBoost | Iris Dataset                   | Classification Accuracy, Confusion Matrix         |
| `pca.ipynb`                   | Unsupervised Dimensionality Reduction        | Principal Component Analysis (PCA)                            | Synthetic Blobs (`make_blobs`) | Explained Variance Ratio                          |
| `unsupervised.ipynb`          | Unsupervised Clustering & Geometric Limits   | K-Means, DBSCAN                                               | `make_blobs`, `make_moons`     | WCSS / Inertia, Cluster Visualizations            |

---

## Detailed Notebook Breakdowns

### 1. `insurance.ipynb` - Regression & Statistical Feature Selection

This notebook implements an end-to-end regression workflow predicting personal medical insurance charges based on demographic and lifestyle indicators.

#### Key Highlights & Workflow

- **Exploratory Data Analysis (EDA):**
  - Distribution analysis of continuous features (`age`, `bmi`, `charges`) using histograms with KDE overlays, identifying significant positive skew in the target variable `charges`.
  - Categorical breakdowns for `sex`, `smoker`, `region`, and `children`.
  - Bivariate scatter and box plots demonstrating the interaction between smoking status, BMI thresholds, and medical costs.
- **Data Cleaning & Encoding:**
  - Identification and removal of duplicate records.
  - Binary encoding for `sex` (`is_female`) and `smoker` (`is_smoker`).
  - One-hot encoding for the `region` attribute with baseline category dropped (`drop_first=True`) to prevent dummy variable trap multicollinearity.
- **Feature Engineering:**
  - Discretization of continuous BMI values into clinical categories: Underweight (`< 18.5`), Normal (`18.5 - 24.9`), Overweight (`25 - 29.9`), and Obese (`>= 30`).
  - Dummy encoding for newly engineered BMI category flags.
- **Feature Scaling:**
  - Z-score standardization using `StandardScaler` on numerical columns (`age`, `bmi`, `children`).
- **Statistical Hypothesis Testing for Feature Selection:**
  - **Pearson Correlation (`scipy.stats.pearsonr`):** Quantified linear dependency between each numerical/encoded predictor and `charges`.
  - **Chi-Square Test of Independence (`scipy.stats.chi2_contingency`):** Discretized the continuous target `charges` into quartiles (`pd.qcut`) and ran cross-tabulation hypothesis testing against categorical attributes at significance level `alpha = 0.05` to formally reject or accept feature relevance.
- **Model Training & Evaluation:**
  - Multiple Linear Regression (`LinearRegression`) on selected feature subsets.
  - Model assessment using both standard $R^2$ and Adjusted $R^2$ to penalize extraneous feature dimensions:
    $$\text{Adjusted } R^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$

---

### 2. `titanic.ipynb` - Classification Benchmark & Data Leakage Prevention

A systematic classification study evaluating multiple predictive models on passenger survival from the Titanic dataset.

#### Key Highlights & Workflow

- **Data Sanitization & Preprocessing:**
  - Dropped redundant and high-cardinality/leaky columns: `deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`.
  - Imputed missing values in `age` using feature mean.
  - Filtered out rows with missing `embarked` values.
  - Converted categorical labels (`sex`, `embarked`) into numerical format via `LabelEncoder`.
- **Data Leakage Mitigation:**
  - Enforced strict train-test separation (80/20 split) prior to standardization.
  - `StandardScaler` was fitted strictly on training data (`X_train`) and transformed across `X_train` and `X_test` independently.
- **Comparative Multi-Model Training:**
  - Evaluated five fundamental classification algorithms under identical feature conditions:
    1. **Logistic Regression:** Linear boundary probabilistic classification.
    2. **K-Nearest Neighbors (KNN):** Non-parametric instance classification with $k=5$.
    3. **Gaussian Naive Bayes:** Probabilistic modeling using class-conditional independence assumptions.
    4. **Decision Tree Classifier:** Recursive information gain partitioning (`random_state=42`).
    5. **Support Vector Machine (SVC):** Non-linear maximum margin classification utilizing Radial Basis Function (RBF) kernel.
- **Model Evaluation & Cross-Validation:**
  - Performance measured via accuracy scores, confusion matrices, precision, recall, and F1-scores.
  - Applied 5-Fold Stratified Cross-Validation (`cross_val_score`) to assess generalization performance and variance across training folds.

---

### 3. `gridAndRandomSearchCV.ipynb` - Hyperparameter Optimization

A practical comparison between manual parameter setting, exhaustive grid search, and stochastic random search on multi-class classification.

#### Key Highlights & Workflow

- **Baseline Modeling:**
  - Trained K-Nearest Neighbors ($k=13$) and Support Vector Classifier (`gamma='auto'`) on the Iris dataset.
  - Evaluated baseline accuracy and detailed classification metrics.
- **Exhaustive Parameter Search (`GridSearchCV`):**
  - Evaluated full Cartesian product over parameter space:
    - Regularization parameter $C \in [1, 10, 20, 30]$
    - Kernel functions: `['rbf', 'linear']`
  - Integrated 5-fold cross-validation across all parameter permutations.
  - Extracted optimal configurations (`best_score_`, `best_params_`).
- **Randomized Parameter Search (`RandomizedSearchCV`):**
  - Performed stochastic subsampling with a fixed evaluation budget (`n_iter=4`) across the identical parameter space.
  - Contrasted computational overhead versus parameter convergence against exhaustive grid search.

---

### 4. `ensemble.ipynb` - Ensemble Architectures: Stacking, Bagging & Boosting

An exploration of modern ensemble paradigms demonstrating how weak and base learners combine to minimize bias and variance.

#### Key Highlights & Workflow

- **Stacking Classifier (Meta-Learning):**
  - **Base Learners (Level 0):** Decision Tree, Support Vector Classifier (`probability=True`), and Logistic Regression.
  - **Meta-Learner (Level 1):** Logistic Regression trained on out-of-fold probability predictions generated through internal 5-fold cross-validation.
  - Demonstrates how heterogeneous decision boundaries combine into a single meta-prediction.
- **Bagging (Bootstrap Aggregation):**
  - Implemented `RandomForestClassifier` with ensemble tree aggregation.
  - Demonstrates variance reduction through parallel bootstrap sampling and randomized feature subset selection.
- **Boosting (Sequential Error Correction):**
  - **AdaBoost (`AdaBoostClassifier`):** Sequentially fits decision stumps, iteratively upweighting previously misclassified data points.
  - **Gradient Boosting (`GradientBoostingClassifier`):** Fits sequential trees on negative gradients (pseudo-residuals) of the loss function with shrinkage / learning rate control.
  - **XGBoost (`XGBClassifier`):** Scalable, regularized gradient boosted trees leveraging second-order Taylor expansion approximations and depth constraints.
- **Evaluation:**
  - Compared test set accuracy and classification reports across all ensemble variants.

---

### 5. `pca.ipynb` - Unsupervised Dimensionality Reduction

Implementation of Principal Component Analysis for dimensionality reduction, feature decorrelation, and low-dimensional projection.

#### Key Highlights & Workflow

- **Synthetic High-Dimensional Data:**
  - Generated 500 samples across 5 continuous dimensions with 3 distinct cluster centers using `make_blobs`.
- **Pre-Processing Requirement:**
  - Standardized features with `StandardScaler`. Highlights the theoretical requirement that PCA relies on variance maximization along orthogonal axes, making it sensitive to unstandardized feature scales.
- **Component Projection:**
  - Computed top 2 principal components (`PC1`, `PC2`) using `PCA(n_components=2)`.
  - Analyzed `explained_variance_ratio_` to quantify the percentage of overall system variance captured by the 2D projection.
- **Latent Space Visualization:**
  - Projected data onto the 2D principal plane, confirming class separability is preserved despite compressing feature space from 5D to 2D.

---

### 6. `unsupervised.ipynb` - Clustering Paradigms & Manifold Geometry

A comparative investigation of distance-based centroid clustering versus density-based spatial clustering, including failure analysis on non-convex manifolds.

#### Key Highlights & Workflow

- **K-Means on Convex/Isotropic Distributions:**
  - Standardized 2D Gaussian cluster data (`make_blobs`).
  - **The Elbow Method:** Iterated cluster count $k$ from 1 through 10, computing Within-Cluster Sum of Squares (WCSS / Inertia) at each step to identify the optimal rate-of-change inflection point.
  - Fitted optimal K-Means model ($k=3$) and visualized cluster boundaries along with centroid coordinates.
- **K-Means Failure Analysis on Non-Convex Geometry:**
  - Tested K-Means on interleaved crescent geometries generated via `make_moons`.
  - Demonstrated how K-Means enforces Voronoi partitioning (spherical assumptions), inherently failing when cluster distributions form non-linear, non-convex manifolds.
- **Density-Based Clustering with DBSCAN:**
  - Applied DBSCAN (Density-Based Spatial Clustering of Applications with Noise) on the identical `make_moons` dataset.
  - Conducted sensitivity analysis across neighborhood distance thresholds (`eps`) and core point density minimums (`min_samples=5`).
  - Successfully partitioned non-linear crescent structures without needing a pre-specified cluster count, demonstrating automatic noise/outlier identification.

---

## Technical Stack & Dependencies

The project relies on core Python scientific libraries:

```text
numpy
pandas
scipy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

---
