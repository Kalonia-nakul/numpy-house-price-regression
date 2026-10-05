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

# Step 2 - compute_iqr_bounds (not yet solved)
# TODO: implement

# Step 3 - clip_columns (not yet solved)
# TODO: implement

# Step 4 - make_ratio_feature (not yet solved)
# TODO: implement

# Step 5 - append_column (not yet solved)
# TODO: implement

# Step 6 - one_hot_encode (not yet solved)
# TODO: implement

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

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

