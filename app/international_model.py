import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

def load_data():
    """Load and preprocess the international astronauts data."""
    df = pd.read_csv('data/international_astronauts.csv')
    return df

def compute_smoothed_priors(df, group_cols, target_col, alpha=1.0):
    """
    Compute smoothed priors for a target variable within groups.
    Uses additive smoothing with alpha to handle groups with few samples.
    """
    # Global mean as the base prior
    global_mean = df[target_col].mean()
    
    # Group statistics
    group_stats = df.groupby(group_cols)[target_col].agg(['count', 'mean']).reset_index()
    
    # Apply smoothing
    smoothed_mean = (group_stats['count'] * group_stats['mean'] + alpha * global_mean) / (group_stats['count'] + alpha)
    group_stats['smoothed_mean'] = smoothed_mean
    
    return group_stats

def prepare_features(df):
    """Prepare features for the residual model using one-hot encoding."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    feature_matrix = encoder.fit_transform(df[['country', 'gender']])
    feature_names = (
        [f"country_{c}" for c in encoder.categories_[0]] +
        [f"gender_{g}" for g in encoder.categories_[1]]
    )
    
    return pd.DataFrame(feature_matrix, columns=feature_names), encoder

def train_model(X, y, params=None):
    """Train a GradientBoostingRegressor on the residuals."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
    
    model = GradientBoostingRegressor(**params)
    model.fit(X, y)
    return model

def load_trained_models():
    """Load the trained international models for prediction."""
    models_dir = Path('models')
    
    models = {}
    for target in ['total_flights', 'total_time']:
        models[target] = {
            'priors': joblib.load(models_dir / f'{target}_priors.joblib'),
            'encoder': joblib.load(models_dir / f'{target}_encoder.joblib'),
            'residual_model': joblib.load(models_dir / f'{target}_residual_model.joblib')
        }
    
    return models

def predict_international(country, gender, models):
    """Make predictions for an international astronaut."""
    predictions = {}
    
    for target in ['total_flights', 'total_time']:
        # Get prior prediction
        prior_stats = models[target]['priors']
        prior_match = prior_stats[
            (prior_stats['country'] == country) & 
            (prior_stats['gender'] == gender)
        ]
        
        if len(prior_match) > 0:
            prior_pred = prior_match['smoothed_mean'].iloc[0]
        else:
            # Fallback to global mean if no exact match
            global_mean = prior_stats['smoothed_mean'].mean()
            prior_pred = global_mean
        
        # Get residual prediction
        encoder = models[target]['encoder']
        residual_model = models[target]['residual_model']
        
        # Prepare features
        features_df = pd.DataFrame({'country': [country], 'gender': [gender]})
        X = encoder.transform(features_df[['country', 'gender']])
        residual_pred = residual_model.predict(X)[0]
        
        # Combine predictions
        final_pred = prior_pred + residual_pred
        predictions[target] = max(0, final_pred)  # Ensure non-negative
    
    return predictions

def train_international_models():
    """Train and save the international astronaut models."""
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading international astronaut data...")
    df = load_data()
    
    # Compute smoothed priors for both target variables
    print("Computing smoothed priors...")
    targets = ['total_flights', 'total_time']
    group_cols = ['country', 'gender']
    priors = {}
    baseline_mae = {}
    final_mae = {}
    r2_scores = {}
    
    for target in targets:
        print(f"\nProcessing {target}...")
        
        # Compute priors
        prior_stats = compute_smoothed_priors(df, group_cols, target)
        priors[target] = prior_stats
        
        # Merge priors back to get baseline predictions
        df_with_priors = df.merge(prior_stats, on=group_cols)
        baseline_preds = df_with_priors['smoothed_mean']
        baseline_mae[target] = mean_absolute_error(df[target], baseline_preds)
        
        # Compute residuals
        residuals = df[target] - baseline_preds
        
        # Prepare features for residual model
        print("Preparing features...")
        X, encoder = prepare_features(df)
        
        # Train residual model
        print("Training residual model...")
        residual_model = train_model(X, residuals)
        
        # Final predictions (priors + residuals)
        residual_preds = residual_model.predict(X)
        final_preds = baseline_preds + residual_preds
        final_mae[target] = mean_absolute_error(df[target], final_preds)
        r2_scores[target] = r2_score(df[target], final_preds)
        
        # Save artifacts
        print("Saving artifacts...")
        joblib.dump(prior_stats, models_dir / f'{target}_priors.joblib')
        joblib.dump(encoder, models_dir / f'{target}_encoder.joblib')
        joblib.dump(residual_model, models_dir / f'{target}_residual_model.joblib')
        
        # Print metrics
        print(f"\nResults for {target}:")
        print(f"Baseline (Priors only) MAE: {baseline_mae[target]:.2f}")
        print(f"Final (Priors + Residuals) MAE: {final_mae[target]:.2f}")
        print(f"Final R² Score: {r2_scores[target]:.4f}")
        improvement = ((baseline_mae[target] - final_mae[target]) / baseline_mae[target] * 100)
        print(f"Improvement: {improvement:.1f}%")
    
    # Save performance metrics summary
    performance_summary = {
        'baseline_mae': baseline_mae,
        'final_mae': final_mae,
        'r2_scores': r2_scores,
        'num_astronauts': len(df),
        'num_countries': df['country'].nunique(),
        'num_genders': df['gender'].nunique()
    }
    joblib.dump(performance_summary, models_dir / 'performance_metrics.joblib')
    
    # Return summary for display
    return performance_summary

if __name__ == "__main__":
    results = train_international_models()
    print("\n" + "="*50)
    print("INTERNATIONAL MODEL TRAINING COMPLETE")
    print("="*50)
    print(f"Total astronauts: {results['num_astronauts']}")
    print(f"Countries represented: {results['num_countries']}")
    print(f"Genders: {results['num_genders']}")
    print("\nFinal Model Performance:")
    for target in ['total_flights', 'total_time']:
        print(f"  {target}: MAE={results['final_mae'][target]:.2f}, R²={results['r2_scores'][target]:.4f}")