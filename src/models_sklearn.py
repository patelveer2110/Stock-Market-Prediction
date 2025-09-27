# src/models_sklearn.py
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def build_svr(C=1.0, epsilon=0.01, kernel='rbf'):
    return SVR(C=C, epsilon=epsilon, kernel=kernel)

def build_rf(n_estimators=200, max_depth=None, n_jobs=1, random_state=42):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=random_state)

def build_xgb(n_estimators=200, learning_rate=0.05, max_depth=6):
    return XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, objective='reg:squarederror', verbosity=0)
