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

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

