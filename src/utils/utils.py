import pandas as pd
import numpy as np
import platform
import psutil
import warnings
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
from scipy.stats.mstats import winsorize
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

def check_apple_silicon():
    """Check if running on Apple Silicon Mac."""
    if platform.system() == 'Darwin':
        if platform.processor() == 'arm' or 'Apple' in platform.processor():
            print("🍎 Apple Silicon detected!")
            print(f"   Processor: {platform.processor()}")
            print(f"   Machine: {platform.machine()}")
            return True
    return False

def get_m3_optimized_params():
    """Get optimized parameters for M3 Mac."""
    cpu_count = psutil.cpu_count(logical=False)
    thread_count = psutil.cpu_count(logical=True)
    print(f"🖥️ M3 Mac Configuration:")
    print(f"   Physical cores: {cpu_count}")
    print(f"   Total threads: {thread_count}")
    print(f"   Available memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    params = {
        'n_jobs': cpu_count,
        'tree_method': 'hist',
        'max_bin': 256,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'predictor': 'cpu_predictor',
        'verbosity': 0,
        'random_state': 42
    }
    return params, cpu_count

def get_m3_lightgbm_params(cpu_count):
    """LightGBM parameters optimized for M3."""
    return {
        'device_type': 'cpu',
        'num_threads': cpu_count,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'force_col_wise': True,
        'histogram_pool_size': -1
    }

def check_mps_availability():
    """Check if PyTorch MPS backend is available."""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("🎮 PyTorch Metal Performance Shaders (MPS) available!")
            print(f"   MPS built: {torch.backends.mps.is_built()}")
            return True
    except:
        pass
    return False

def load_data_optimized(filepath, chunksize=100000):
    """Load data optimized for M3's unified memory architecture."""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            if df[col].min() >= 0 and df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    return df

def treat_outliers(X, outlier_percentile=0.005):
    X_out = X.copy()
    for col in X_out.columns:
        z_scores = np.abs(stats.zscore(X_out[col].astype(float)))
        extreme_outliers = (z_scores > 4).sum()
        if extreme_outliers > 5:
            X_out[col] = winsorize(X_out[col], limits=(outlier_percentile, outlier_percentile))
    return X_out

def impute_missing(X):
    X_out = X.copy()
    for col in X_out.columns:
        if X_out[col].isnull().sum() > 0:
            fill_value = X_out[col].median() if X_out[col].dtype in ['int64', 'float64', 'float32', 'int32'] else X_out[col].mode()[0]
            X_out[col].fillna(fill_value, inplace=True)
    return X_out

def remove_low_variance_features(X, threshold=0.001):
    return X.loc[:, X.var() >= threshold]

def ensemble_feature_ranking(X, y, top_n=20, random_state=42):
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    f_scores, _ = f_classif(X, y)
    xgb = XGBClassifier(random_state=random_state, verbosity=0)
    xgb.fit(X, y)
    xgb_importance = xgb.feature_importances_
    feature_rankings = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi_scores,
        'F_Score': f_scores,
        'XGB_Importance': xgb_importance
    })
    for col in ['MI_Score', 'F_Score', 'XGB_Importance']:
        feature_rankings[f'{col}_norm'] = (
            (feature_rankings[col] - feature_rankings[col].min()) /
            (feature_rankings[col].max() - feature_rankings[col].min() + 1e-9)
        )
    feature_rankings['Ensemble_Score'] = (
        feature_rankings['MI_Score_norm'] +
        feature_rankings['F_Score_norm'] +
        feature_rankings['XGB_Importance_norm']
    ) / 3
    feature_rankings = feature_rankings.sort_values('Ensemble_Score', ascending=False)
    top_features = feature_rankings.head(top_n)['Feature'].tolist()
    return top_features, feature_rankings

def conservative_normalization(X):
    X_norm = X.copy()
    for col in X_norm.columns:
        mean_val = X_norm[col].mean()
        std_val = X_norm[col].std()
        if std_val > 0:
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            X_norm[col] = np.clip(X_norm[col], lower_bound, upper_bound)
    return X_norm

def intelligent_sampling(df, target='buy', target_size=200000, strategy='balanced', random_state=42):
    original_size = len(df)
    if strategy == 'balanced':
        samples_per_class = target_size // 2
        buy_samples = df[df[target] == True].sample(
            n=min(samples_per_class, (df[target] == True).sum()),
            random_state=random_state
        )
        no_buy_samples = df[df[target] == False].sample(
            n=min(samples_per_class, (df[target] == False).sum()),
            random_state=random_state
        )
        sampled_df = pd.concat([buy_samples, no_buy_samples], ignore_index=True)
    elif strategy == 'stratified':
        sampled_df = df.sample(
            n=min(target_size, len(df)),
            random_state=random_state,
            stratify=df[target] if target in df.columns else None
        )
    elif strategy == 'recent':
        sampled_df = df.tail(target_size)
    elif strategy == 'diverse':
        sampled_df = df.sample(
            n=min(target_size, len(df)),
            random_state=random_state
        )
    else:
        sampled_df = df.sample(
            n=min(target_size, len(df)),
            random_state=random_state
        )
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df

class M3OptimizedPipeline:
    """Pipeline optimized for M3 Mac with 1.4M samples."""
    def __init__(self):
        self.is_m3 = check_apple_silicon()
        self.m3_params, self.cpu_count = get_m3_optimized_params()
        self.has_mps = check_mps_availability()
    def optimize_sampling(self, df, target_size=300000):
        print(f"\n🎯 M3-OPTIMIZED SAMPLING")
        print("-" * 40)
        original_size = len(df)
        if original_size <= target_size:
            return df
        target = 'buy'
        buy_mask = df[target] == True
        buy_samples = df[buy_mask]
        no_buy_samples = df[~buy_mask]
        n_buy = len(buy_samples)
        n_no_buy = len(no_buy_samples)
        if n_buy < target_size // 2:
            buy_sampled = buy_samples.sample(
                n=target_size // 2,
                replace=True,
                random_state=42
            )
            no_buy_sampled = no_buy_samples.sample(
                n=target_size // 2,
                random_state=42
            )
        else:
            buy_sampled = buy_samples.sample(
                n=target_size // 2,
                random_state=42
            )
            no_buy_sampled = no_buy_samples.sample(
                n=target_size // 2,
                random_state=42
            )
        sampled_df = pd.concat([buy_sampled, no_buy_sampled])
        sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   ✅ Sampling complete:")
        print(f"      Original: {original_size:,} samples")
        print(f"      Sampled: {len(sampled_df):,} samples")
        print(f"      Speedup: {original_size / len(sampled_df):.1f}x")
        return sampled_df
    def train_models_parallel(self, X_train, y_train, X_val, y_val, best_params=None):
        import xgboost as xgb
        from xgboost import XGBClassifier
        import lightgbm as lgb
        print(f"\n🚀 M3-OPTIMIZED MODEL TRAINING")
        print("-" * 40)
        results = {}
        print("   🌲 Training XGBoost...")
        import time
        start_time = time.time()
        if best_params is not None:
            xgb_params = {**best_params, 'early_stopping_rounds': 30}
        else:
            xgb_params = {
                **self.m3_params,
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.1,
                'early_stopping_rounds': 30
            }
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        xgb_time = time.time() - start_time
        from sklearn.metrics import roc_auc_score
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        results['XGBoost'] = {
            'model': xgb_model,
            'time': xgb_time,
            'auc': xgb_auc
        }
        print(f"      ✅ XGBoost: AUC={xgb_auc:.4f}, Time={xgb_time:.1f}s")
        print("   🌲 Training LightGBM...")
        start_time = time.time()
        lgb_params = get_m3_lightgbm_params(self.cpu_count)
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        lgb_time = time.time() - start_time
        lgb_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        lgb_auc = roc_auc_score(y_val, lgb_pred)
        results['LightGBM'] = {
            'model': lgb_model,
            'time': lgb_time,
            'auc': lgb_auc
        }
        print(f"      ✅ LightGBM: AUC={lgb_auc:.4f}, Time={lgb_time:.1f}s")
        print(f"\n   📊 M3 Performance Comparison:")
        print(f"      XGBoost: {X_train.shape[0] / xgb_time:.0f} samples/second")
        print(f"      LightGBM: {X_train.shape[0] / lgb_time:.0f} samples/second")
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        print(f"\n   🏆 Best model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
        return results[best_model_name]['model'], results
    def optimize_hyperparameters_fast(self, X_train, y_train, X_val, y_val, time_budget=180):
        import xgboost as xgb
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score
        print(f"\n🔍 M3-OPTIMIZED HYPERPARAMETER SEARCH")
        print("-" * 45)
        param_combinations = [
            {'max_depth': 4, 'learning_rate': 0.15, 'n_estimators': 200},
            {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300},
            {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 400},
            {'max_depth': 6, 'learning_rate': 0.15, 'n_estimators': 250},
            {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 350},
        ]
        best_params = None
        best_auc = 0
        subset_size = min(50000, len(X_train))
        if subset_size < len(X_train):
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_subset = X_train.iloc[indices]
            y_subset = y_train.iloc[indices]
        else:
            X_subset = X_train
            y_subset = y_train
        for params in param_combinations:
            all_params = {**self.m3_params, **params}
            model = XGBClassifier(**all_params)
            model.fit(X_subset, y_subset, verbose=False)
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            if auc > best_auc:
                best_auc = auc
                best_params = all_params
            print(f"   Params: {params} → AUC: {auc:.4f}")
        print(f"\n   ✅ Best params found: AUC={best_auc:.4f}")
        return best_params 