import pandas as pd
import numpy as np
import pickle

# 1. Load Data
df = pd.read_csv('house_data.csv')

# --- 2. ADVANCED FEATURE ENGINEERING ---
# A. Geometry & Interactions: The value of space changes based on quality (grade)
df['log_sqft_living'] = np.log(df['sqft_living'])
df['living_grade_interaction'] = df['log_sqft_living'] * df['grade']

# B. Location Intelligence: Euclidean distance to Seattle City Center
# Latitude: 47.6062, Longitude: -122.3321
df['dist_to_center'] = np.sqrt((df['lat'] - 47.6062)**2 + (df['long'] - (-122.3321))**2)

# C. Basic Time Features
df['house_age'] = 2015 - df['yr_built']
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

# D. Encode Zipcodes (High-precision location data)
zip_dummies = pd.get_dummies(df['zipcode'], prefix='zip').astype(float)

# Select features
num_features = ['bedrooms', 'bathrooms', 'log_sqft_living', 'living_grade_interaction',
                'dist_to_center', 'grade', 'waterfront', 'view', 'condition', 'house_age']

X_df = pd.concat([df[num_features], zip_dummies], axis=1)

# Target: Log-transform price to handle skewness
y = np.log(df['price'].values).reshape(-1, 1)
X = X_df.values

# Add Intercept (Column of 1s)
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

# 3. Manual Train/Test Split
split_idx = int(0.75 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- 4. MANUAL RIDGE REGRESSION ---
# Adding alpha (lambda) regularizes the weights, preventing overfitting.
# Formula: beta = (X^T * X + alpha * I)^-1 * X^T * y
alpha = 0.1
I = np.eye(X_train.shape[1])
I[0, 0] = 0  # Do not regularize the intercept term

xt_x_reg = np.dot(X_train.T, X_train) + alpha * I
weights = np.dot(np.linalg.inv(xt_x_reg), np.dot(X_train.T, y_train))

# 5. Predict and Back-transform
y_pred_log = np.dot(X_test, weights)
y_pred = np.exp(y_pred_log) # Convert from Log-Price back to Dollars
y_test_actual = np.exp(y_test)

# 6. Evaluation
ss_res = np.sum((y_test_actual - y_pred) ** 2)
ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
r2_score = 1 - (ss_res / ss_tot)
mae = np.mean(np.abs(y_test_actual - y_pred))

print(f"Final Ridge R-squared: {r2_score:.4f}")
print(f"Final Ridge MAE: ${mae:,.2f}")

# 7. Save Model
with open('ridge_house_model.pkl', 'wb') as f:
    pickle.dump({'weights': weights, 'features': X_df.columns.tolist()}, f)
     