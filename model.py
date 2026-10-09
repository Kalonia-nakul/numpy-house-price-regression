"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    X = np.array(X, dtype=float)      # X: (N rows, F columns) copy, float
    N, F = X.shape
    for i in range(F):                # i = column index
        total = 0.0                   # sum of non-NaN values in column i
        count = 0                     # how many non-NaN values in column i
        for j in range(N):            # j = row index
            if not np.isnan(X[j, i]):
                total += X[j, i]
                count += 1
        mean = total / count if count > 0 else 0.0
        for j in range(N):
            if np.isnan(X[j, i]):
                X[j, i] = mean
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
   
    q1, q3 = np.percentile(X, [25, 75], axis=0)     
    iqr = q3 - q1                
    lower = q1 - k * iqr                          
    upper = q3 + k * iqr 
    return lower , upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    X = np.array(X, dtype=float)          # np.array copies by default; X is now a new array
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = min(max(X[i][j] , lower[j]) , upper[j])
    return X

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    r = [0] * len(numerator)
  
    r = numerator / (denominator + eps)
    return r

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack((X,col))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    categories = np.unique(labels)
    
    # Create an output matrix filled with zeros
    encoded = np.zeros((len(labels), len(categories)))
    
    # Put 1 at the position corresponding to each category
    for i, value in enumerate(labels):
        index = np.where(categories == value)[0][0]
        encoded[i, index] = 1.0
    
    return encoded

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Replace zero standard deviations with 1
    std[std == 0] = 1.0

    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    X = np.asarray(X, dtype=float)        # X: (N, F) data
    mean = np.asarray(mean, dtype=float)  # mean: (F,) per-column mean
    std = np.asarray(std, dtype=float)    # std: (F,) per-column standard deviation
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    X = np.asarray(X, dtype=float)            
    ones = np.ones((X.shape[0], 1))            
    return np.hstack([ones, X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng = np.random.default_rng(seed)   # rng: random generator fixed by the seed
    return rng.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    indices = np.asarray(indices)              
    N = len(indices)                           
    n_train = int(train_ratio * N)              
    n_val = int(val_ratio * N)                  
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]        
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    idx = np.asarray(indices)        # idx: new local variable, no clash with the parameter
    return np.asarray(X)[idx], np.asarray(y)[idx]

# Step 13 - ols_fit
def ols_fit(X, y):
    X = np.asarray(X, dtype=float)                 
    y = np.asarray(y, dtype=float)                   
    w, *_ = np.linalg.lstsq(X, y, rcond=None)        
    return w

# Step 14 - ols_predict
def ols_predict(X, theta):
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    y_true = np.asarray(y_true, dtype=float)             # y_true: (N,) true targets
    y_pred = np.asarray(y_pred, dtype=float)             # y_pred: (N,) predictions
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_true = np.asarray(y_true, dtype=float)         # y_true: (N,) true targets
    y_pred = np.asarray(y_pred, dtype=float)         # y_pred: (N,) predictions
    ss_res = np.sum((y_true - y_pred) ** 2)          # ss_res: total squared error
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)   # ss_tot: total squared spread around the mean
    if ss_tot == 0:                                  # constant y_true: R^2 undefined, return 0.0
        return 0.0
    return float(1 - ss_res / ss_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    y_true = np.asarray(y_true, dtype=float)    # y_true: (N,) true targets
    y_pred = np.asarray(y_pred, dtype=float)    # y_pred: (N,) predictions
    r = y_true - y_pred                         # r: (N,) residuals

    return {
        'mean': float(np.mean(r)),                  # average residual (bias)
        'std': float(np.std(r)),                    # population std (divides by N)
        'median_abs': float(np.median(np.abs(r))),  # median absolute residual
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = np.array(X, dtype=float)                      # X: (N, F) copy, input not modified

    # Step 1: impute NaN with column mean (all-NaN column -> 0)
    counts = np.sum(~np.isnan(X), axis=0)             # counts: (F,) non-NaN count per column
    sums = np.nansum(X, axis=0)                       # sums: (F,) sum of non-NaN per column
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)  # means: (F,) mu_j
    rows, cols = np.where(np.isnan(X))                # positions of NaNs
    X[rows, cols] = means[cols]                       # fill each NaN with its column mean

    # Step 2: IQR bounds per column, from the imputed data
    q1, q3 = np.percentile(X, [25, 75], axis=0)       # q1, q3: (F,) quartiles
    iqr = q3 - q1                                     # iqr: (F,) spread
    lower = q1 - iqr_k * iqr                          # lower: (F,) l_j
    upper = q3 + iqr_k * iqr                          # upper: (F,) u_j

    # Step 3: clip each column to its own bounds
    return np.clip(X, lower, upper)

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    X = np.asarray(X_num, dtype=float)                            # X: (N, F) numeric features
    ratio = (X[:, ratio_num_idx] / X[:, ratio_den_idx]).reshape(-1, 1)  # ratio: (N, 1), q_i
    out = np.hstack([X, ratio])                                   # (N, F+1)

    if cat_labels is not None:
        labels = np.asarray(cat_labels)                           # labels: (N,) category per sample
        cats = np.unique(labels)                                  # cats: (K,) sorted distinct labels
        onehot = (labels[:, None] == cats[None, :]).astype(float) # onehot: (N, K)
        out = np.hstack([out, onehot])                            # (N, F+1+K)
    return out

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    X = np.asarray(X)                                  # X: (N, F) features
    y = np.asarray(y)                                  # y: (N,) targets
    N = len(X)                                         # N: number of samples

    rng = np.random.RandomState(seed)                  # legacy generator fixed by the seed
    perm = rng.permutation(N)                          # perm: (N,) shuffled indices (pi)

    n_train = int(train_ratio * N)                      # size of train part
    n_val = int(val_ratio * N)                          # size of validation part

    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]                        # remainder

    return {
        'X_train': X[tr], 'y_train': y[tr],
        'X_val': X[va],   'y_val': y[va],
        'X_test': X[te],  'y_test': y[te],
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    X_train = np.asarray(splits['X_train'], dtype=float)   # X_train: (n, F) training features
    mean = X_train.mean(axis=0)                            # mean: (F,) mu_j, from train only
    std = X_train.std(axis=0)                              # std: (F,) raw sigma_j (divides by n)
    safe_std = np.where(std == 0, 1.0, std)                # safe_std: (F,) sigma'_j, 1.0 for constant columns

    out = dict(splits)                                     # copy dict; y_* entries carry over unchanged
    for name in ('X_train', 'X_val', 'X_test'):
        X = np.asarray(splits[name], dtype=float)          # X: (m, F) this part's features
        Z = (X - mean) / safe_std                          # Z: (m, F) standardized with train stats
        ones = np.ones((Z.shape[0], 1))                    # ones: (m, 1) bias column
        out[name] = np.hstack([ones, Z])                   # (m, F+1), bias first

    return out, mean, safe_std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    y_true = np.asarray(y_true, dtype=float)          # y_true: (N,) true targets
    y_pred = np.asarray(y_pred, dtype=float)          # y_pred: (N,) predictions
    r = y_true - y_pred                               # r: (N,) residuals

    ss_res = np.sum(r ** 2)                           # total squared error
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)    # total squared spread around the mean
    r2 = 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    return {
        'mae': float(np.mean(np.abs(r))),
        'rmse': float(np.sqrt(np.mean(r ** 2))),
        'r2': r2,
        'residual_summary': {
            'mean': float(np.mean(r)),
            'std': float(np.std(r)),
            'median_abs': float(np.median(np.abs(r))),
        },
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X = np.asarray(X, dtype=float)                       # X: (N, F) raw features
    y = np.asarray(y, dtype=float)                       # y: (N,) targets

    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)   # impute + IQR clip
    X_all = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels)

    splits = make_train_val_test(X_all, y, train_ratio, val_ratio, seed)
    std_splits, mean, std = standardize_and_add_bias(splits)

    theta = ols_fit(std_splits['X_train'], std_splits['y_train'])   # theta: (P,) weights

    y_val_pred = ols_predict(std_splits['X_val'], theta)
    y_test_pred = ols_predict(std_splits['X_test'], theta)

    return {
        'theta': theta,
        'val_metrics': evaluate_predictions(std_splits['y_val'], y_val_pred),
        'test_metrics': evaluate_predictions(std_splits['y_test'], y_test_pred),
        'y_test': std_splits['y_test'],
        'y_test_pred': y_test_pred,
    }

