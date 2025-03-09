import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load the dataset and prepare it for analysis
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Convert any remaining object columns to numeric if possible
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_target_variable(df, horizon=4, threshold=0.0):
    """
    Create a binary target variable indicating price direction after 'horizon' periods
    """
    # Calculate future price change
    future_price = df['close'].shift(-horizon)
    price_change_pct = (future_price - df['close']) / df['close'] * 100

    # Generate binary labels: 1 for up, 0 for down/sideways
    labels = (price_change_pct > threshold).astype(int)
    print(f"Created labels: {labels.sum()} ups, {len(labels) - labels.sum()} downs")
    print(f"Class balance: {labels.mean() * 100:.2f}% up")

    return labels


def clean_dataset(df):
    """
    Clean the dataset by handling inf, NaN, and extreme values
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Replace inf/-inf with NaN
    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with column medians (more robust than mean)
    for col in cleaned_df.columns:
        if cleaned_df[col].isna().any():
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)

    # Identify and cap extreme outliers using quantiles
    # (only for non-price columns to preserve price information)
    price_cols = ['open', 'high', 'low', 'close']
    for col in cleaned_df.columns:
        if col not in price_cols:
            q1 = cleaned_df[col].quantile(0.01)
            q3 = cleaned_df[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 5 * iqr
            upper_bound = q3 + 5 * iqr

            # Cap extreme values
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)

    return cleaned_df


def correlation_analysis(df, target_col='close', threshold=0.7, top_n=20):
    """
    Analyze correlations between features and with the target
    """
    print("\n=== CORRELATION ANALYSIS ===")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    # Get correlations with target column
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
    print(f"\nTop {top_n} features correlated with {target_col}:")
    print(target_corr.head(top_n + 1))  # +1 to include target itself

    # Identify highly correlated feature pairs (for potential elimination)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\nFound {len(high_corr_pairs)} pairs of features with correlation > {threshold}")
    print("\nTop 10 highly correlated feature pairs (potential redundancy):")
    for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:10]):
        print(f"{i + 1}. {feat1} & {feat2}: {corr:.4f}")

    return target_corr, high_corr_pairs, corr_matrix


def feature_stability_analysis(df, window_size=100, top_n=20):
    """
    Analyze the stability of features over time using coefficient of variation
    """
    print("\n=== FEATURE STABILITY ANALYSIS ===")

    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate coefficient of variation (CV) for each feature
    cv_values = {}
    for column in numeric_df.columns:
        mean_val = abs(numeric_df[column].mean())
        # Avoid division by zero or very small numbers
        if mean_val > 1e-6:
            cv = numeric_df[column].std() / mean_val
            # Check for reasonable CV values
            if np.isfinite(cv) and cv < 1e6:  # Filter out unreasonably large CVs
                cv_values[column] = cv

    # Convert to DataFrame and sort
    cv_df = pd.DataFrame(list(cv_values.items()), columns=['Feature', 'CV'])
    cv_df = cv_df.sort_values('CV')

    print("\nMost stable features (lowest coefficient of variation):")
    print(cv_df.head(top_n))

    print("\nLeast stable features (highest coefficient of variation):")
    print(cv_df.tail(top_n))

    # Calculate rolling CV to check stability over time for key features
    rolling_cv = {}
    key_features = ['close', 'volume', 'RSI_14'] + [col for col in numeric_df.columns if 'MA_4' in col][:3]
    key_features = [f for f in key_features if f in numeric_df.columns]

    for feature in key_features:
        rolling_mean = numeric_df[feature].rolling(window=window_size).mean()
        rolling_std = numeric_df[feature].rolling(window=window_size).std()
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_series = rolling_std / rolling_mean.abs()
            # Replace inf and NaN with NaN to properly calculate mean later
            cv_series = cv_series.replace([np.inf, -np.inf], np.nan)
            rolling_cv[feature] = cv_series

    # Calculate average CV over all rolling windows
    avg_rolling_cv = {}
    for feature, values in rolling_cv.items():
        avg_rolling_cv[feature] = values.mean()

    # Sort by average rolling CV
    sorted_rolling_cv = sorted(avg_rolling_cv.items(), key=lambda x: x[1])

    print("\nAverage rolling CV for key features:")
    for feature, avg_cv in sorted_rolling_cv:
        print(f"{feature}: {avg_cv:.4f}")

    return cv_df, rolling_cv


def analyze_autocorrelation(df, columns=None, lags=10):
    """
    Analyze autocorrelation in time series to identify temporal dependencies
    """
    print("\n=== AUTOCORRELATION ANALYSIS ===")

    if columns is None:
        columns = ['close', 'volume']

    results = {}
    for col in columns:
        if col in df.columns:
            # Calculate autocorrelation for different lags
            autocorr = [df[col].autocorr(lag=i) for i in range(1, lags + 1)]
            results[col] = autocorr

            print(f"\nAutocorrelation for {col}:")
            for i, ac in enumerate(autocorr):
                print(f"Lag {i + 1}: {ac:.4f}")

    return results


def pca_analysis(df, n_components=20):
    """
    Use PCA to identify important feature dimensions
    """
    print("\n=== PRINCIPAL COMPONENT ANALYSIS ===")

    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Use robust scaler which is less sensitive to outliers
    scaler = RobustScaler()

    try:
        # Scale the data
        scaled_data = scaler.fit_transform(numeric_df)

        # Apply PCA
        n_components = min(n_components, numeric_df.shape[1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)

        # Analyze explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        print("\nExplained variance by principal components:")
        for i, var in enumerate(explained_variance[:10]):  # Show first 10
            print(f"PC{i + 1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

        # Find number of components needed for 90% variance
        n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
        print(f"\n{n_components_90} components needed to explain 90% of variance")

        # Calculate feature contributions to principal components
        feature_importance = {}

        for i, feature in enumerate(numeric_df.columns):
            # Sum of absolute loadings across components, weighted by explained variance
            importance = sum(abs(pca.components_[j, i]) * explained_variance[j]
                             for j in range(min(10, n_components)))  # Use top 10 components
            feature_importance[feature] = importance

        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(feature_importance.items(),
                                                     key=lambda item: item[1], reverse=True)}

        print("\nTop 20 features by PCA importance:")
        for i, (feature, importance) in enumerate(list(sorted_importance.items())[:20]):
            print(f"{i + 1}. {feature}: {importance:.6f}")

        return sorted_importance, pca, explained_variance

    except Exception as e:
        print(f"Error in PCA analysis: {str(e)}")
        # Return empty dict if PCA fails
        return {}, None, []


def mutual_information_analysis(df, target_col='close', top_n=20):
    """
    Calculate mutual information between features and target
    """
    print("\n=== MUTUAL INFORMATION ANALYSIS ===")

    try:
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Extract target column
        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]

        # Calculate mutual information
        mi_scores = {}
        for column in X.columns:
            try:
                mi = mutual_info_regression(
                    X[[column]],  # Use 2D array for scikit-learn
                    y,
                    random_state=42
                )[0]

                # Only include valid MI scores
                if np.isfinite(mi) and not np.isnan(mi):
                    mi_scores[column] = mi
            except Exception as column_error:
                print(f"Skipping column {column} in MI calculation: {str(column_error)}")

        # Sort by MI score
        sorted_mi = {k: v for k, v in sorted(mi_scores.items(),
                                             key=lambda item: item[1], reverse=True)}

        print(f"\nTop {top_n} features by mutual information with {target_col}:")
        for i, (feature, mi) in enumerate(list(sorted_mi.items())[:top_n]):
            print(f"{i + 1}. {feature}: {mi:.6f}")

        return sorted_mi

    except Exception as e:
        print(f"Error in mutual information analysis: {str(e)}")
        return {}


def analyze_feature_groups(df, target_corr, pca_importance, mi_scores=None):
    """
    Analyze features by category/group
    """
    print("\n=== FEATURE GROUP ANALYSIS ===")

    # Define feature groups based on naming patterns
    feature_groups = {
        'Price': [col for col in df.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close'])
                  and not any(x in col for x in ['MA', 'SMA', 'EMA', 'BB', 'BAND', 'RSI', 'MACD'])],
        'Volume': [col for col in df.columns if 'volume' in col.lower()],
        'Moving Averages': [col for col in df.columns if any(x in col for x in ['MA_', 'SMA', 'EMA', 'WMA', 'HMA'])],
        'Bollinger Bands': [col for col in df.columns if any(x in col for x in ['BB_', 'BAND'])],
        'RSI': [col for col in df.columns if 'RSI' in col],
        'MACD': [col for col in df.columns if 'MACD' in col],
        'Market Features': [col for col in df.columns if 'MARKET_FEATURES' in col],
        'Horizon Features': [col for col in df.columns if 'HORIZON_ALIGNED' in col]
    }

    # Calculate average importance metrics for each group
    group_stats = {}
    for group_name, columns in feature_groups.items():
        if columns:
            # Filter to existing columns only
            valid_columns = [col for col in columns if col in df.columns]
            if not valid_columns:
                continue

            # Calculate average correlation with target
            corr_values = [target_corr.get(col, 0) for col in valid_columns if col in target_corr]
            avg_corr = np.mean(corr_values) if corr_values else 0

            # Calculate average PCA importance
            pca_values = [pca_importance.get(col, 0) for col in valid_columns if col in pca_importance]
            avg_pca = np.mean(pca_values) if pca_values else 0

            # Calculate average MI score if available
            avg_mi = None
            if mi_scores:
                mi_values = [mi_scores.get(col, 0) for col in valid_columns if col in mi_scores]
                avg_mi = np.mean(mi_values) if mi_values else 0

            # Get top features in this group by correlation
            top_by_corr = sorted([(col, target_corr.get(col, 0)) for col in valid_columns if col in target_corr],
                                 key=lambda x: x[1], reverse=True)[:3]

            # Get top features in this group by PCA importance
            top_by_pca = sorted([(col, pca_importance.get(col, 0)) for col in valid_columns if col in pca_importance],
                                key=lambda x: x[1], reverse=True)[:3]

            # Store stats
            group_stats[group_name] = {
                'count': len(valid_columns),
                'avg_correlation': avg_corr,
                'avg_pca_importance': avg_pca,
                'avg_mi': avg_mi,
                'top_by_correlation': top_by_corr,
                'top_by_pca': top_by_pca
            }

    # Print stats for each group
    for group_name, stats in sorted(group_stats.items(), key=lambda x: x[1]['avg_correlation'], reverse=True):
        print(f"\n{group_name} ({stats['count']} features):")
        print(f"  Average correlation with target: {stats['avg_correlation']:.4f}")
        print(f"  Average PCA importance: {stats['avg_pca_importance']:.6f}")

        if stats['avg_mi'] is not None:
            print(f"  Average mutual information: {stats['avg_mi']:.6f}")

        print("  Top features by correlation:")
        for i, (feat, corr) in enumerate(stats['top_by_correlation'][:3]):
            print(f"    {i + 1}. {feat}: {corr:.4f}")

        print("  Top features by PCA importance:")
        for i, (feat, imp) in enumerate(stats['top_by_pca'][:3]):
            print(f"    {i + 1}. {feat}: {imp:.6f}")

    return feature_groups, group_stats


def calculate_composite_scores(df, target_corr, pca_importance, cv_df, mi_scores=None):
    """
    Calculate a composite score for feature selection
    """
    print("\n=== COMPOSITE FEATURE SCORES ===")

    # Create a dictionary to hold all scores
    scores = {}

    # Convert CV DataFrame to dictionary
    cv_dict = dict(zip(cv_df['Feature'], cv_df['CV']))

    # Define core price features
    core_features = ['open', 'high', 'low', 'close', 'volume']

    # Find max values for normalization
    pca_max = max(pca_importance.values()) if pca_importance else 1
    mi_max = max(mi_scores.values()) if mi_scores and mi_scores else 1

    # Calculate scores for each feature
    for feature in df.select_dtypes(include=[np.number]).columns:
        # Base score (higher for core features)
        base_score = 10 if feature in core_features else 1

        # Correlation score (scale from 0-5)
        corr_score = target_corr.get(feature, 0) * 5

        # PCA importance score (scale from 0-5)
        pca_score = pca_importance.get(feature, 0) / pca_max * 5 if pca_max > 0 else 0

        # Stability score (inverse of CV, scale from 0-5)
        stability_score = 0
        if feature in cv_dict:
            max_stable_cv = 1.0  # Threshold for maximum stability score
            if cv_dict[feature] <= max_stable_cv:
                stability_score = (max_stable_cv - cv_dict[feature]) / max_stable_cv * 5

        # Mutual information score (if available)
        mi_score = 0
        if mi_scores and feature in mi_scores:
            mi_score = mi_scores.get(feature, 0) / mi_max * 5 if mi_max > 0 else 0

        # Final score
        total_score = base_score + corr_score + pca_score + stability_score + mi_score

        # Store in dictionary
        scores[feature] = {
            'total': total_score,
            'base': base_score,
            'correlation': corr_score,
            'pca': pca_score,
            'stability': stability_score,
            'mi': mi_score
        }

    # Sort by total score
    sorted_scores = {k: v for k, v in sorted(scores.items(),
                                             key=lambda item: item[1]['total'], reverse=True)}

    # Print top features
    print("\nTop 30 features by composite score:")
    for i, (feature, score_dict) in enumerate(list(sorted_scores.items())[:30]):
        components = f"(Base: {score_dict['base']:.1f}, Corr: {score_dict['correlation']:.1f}, PCA: {score_dict['pca']:.1f}, Stability: {score_dict['stability']:.1f}"
        if mi_scores:
            components += f", MI: {score_dict['mi']:.1f}"
        components += ")"
        print(f"{i + 1}. {feature}: {score_dict['total']:.2f} {components}")

    return sorted_scores


def select_best_features(df, sorted_scores, feature_groups, n_per_group=2, n_total=30):
    """
    Select the best features using a mixed strategy: top overall + top per group
    """
    print("\n=== FINAL FEATURE SELECTION ===")

    # Start with core price features
    core_features = ['open', 'high', 'low', 'close', 'volume']
    selected_features = [f for f in core_features if f in df.columns]

    print(f"\nSelected core price features ({len(selected_features)}):")
    for feature in selected_features:
        if feature in sorted_scores:
            score = sorted_scores[feature]['total']
            print(f"  - {feature}: {score:.2f}")
        else:
            print(f"  - {feature}")

    # Select top n features from each group
    group_features = []
    for group_name, columns in feature_groups.items():
        if not columns or group_name == 'Price':  # Skip empty groups and Price (already handled)
            continue

        # Get valid columns that exist in our scores
        valid_columns = [col for col in columns if col in sorted_scores]
        if not valid_columns:
            continue

        # Sort by total score
        top_in_group = sorted(valid_columns,
                              key=lambda col: sorted_scores[col]['total'],
                              reverse=True)[:n_per_group]

        print(f"\nTop {n_per_group} features from {group_name}:")
        for feature in top_in_group:
            score = sorted_scores[feature]['total']
            print(f"  - {feature}: {score:.2f}")
            group_features.append(feature)

    # Add group features to selection
    selected_features.extend([f for f in group_features if f not in selected_features])

    # If we still need more features, add top overall that aren't already selected
    top_overall = list(sorted_scores.keys())
    for feature in top_overall:
        if feature not in selected_features and len(selected_features) < n_total:
            selected_features.append(feature)

    # Final selection
    print(f"\nFinal feature selection ({len(selected_features)} features):")
    for i, feature in enumerate(selected_features):
        if feature in sorted_scores:
            score = sorted_scores[feature]['total']
            print(f"{i + 1}. {feature}: {score:.2f}")
        else:
            print(f"{i + 1}. {feature}")

    return selected_features


def remove_redundant_features(df, high_corr_pairs, target_corr, threshold=0.95):
    """
    Remove highly redundant features based on correlation
    """
    print("\n=== REDUNDANT FEATURE REMOVAL ===")

    # Identify extremely correlated pairs (threshold even higher than original correlation threshold)
    redundant_pairs = [(f1, f2, corr) for f1, f2, corr in high_corr_pairs if abs(corr) > threshold]

    # Track which features to remove
    to_remove = set()

    print(f"Found {len(redundant_pairs)} extremely redundant pairs (correlation > {threshold})")

    # For each redundant pair, keep the one most correlated with target
    for feat1, feat2, corr in redundant_pairs:
        # Skip if one of them is already marked for removal
        if feat1 in to_remove or feat2 in to_remove:
            continue

        # Get correlation with target
        corr1 = target_corr.get(feat1, 0)
        corr2 = target_corr.get(feat2, 0)

        # Keep the feature with higher correlation with target
        if corr1 >= corr2:
            to_remove.add(feat2)
            print(f"Removing {feat2} (keeping {feat1})")
        else:
            to_remove.add(feat1)
            print(f"Removing {feat1} (keeping {feat2})")

    # Create a copy of df without the redundant features
    reduced_df = df.drop(columns=list(to_remove))

    print(
        f"Removed {len(to_remove)} redundant features. Dataset reduced from {df.shape[1]} to {reduced_df.shape[1]} columns.")

    return reduced_df, list(to_remove)


def feature_selection_pipeline(df, target_col='close', n_features=30):
    """
    Run the complete feature selection pipeline
    """
    # 0. Clean dataset by handling inf, NaN, and extreme values
    df_cleaned = clean_dataset(df)
    print(f"Dataset cleaned. Original shape: {df.shape}, Cleaned shape: {df_cleaned.shape}")

    # 1. Remove extremely redundant features
    target_corr_initial, high_corr_pairs_initial, _ = correlation_analysis(df_cleaned, target_col)
    df_reduced, removed_features = remove_redundant_features(df_cleaned, high_corr_pairs_initial, target_corr_initial)

    # Continue with reduced dataset

    # 2. Correlation Analysis
    target_corr, high_corr_pairs, corr_matrix = correlation_analysis(df_reduced, target_col)

    # 3. Feature Stability Analysis
    cv_df, rolling_cv = feature_stability_analysis(df_reduced)

    # 4. Autocorrelation Analysis
    autocorr_results = analyze_autocorrelation(df_reduced)

    # 5. PCA Analysis
    pca_importance, pca, explained_variance = pca_analysis(df_reduced)

    # 6. Mutual Information Analysis (if target is provided)
    mi_scores = None
    if target_col in df_reduced.columns:
        mi_scores = mutual_information_analysis(df_reduced, target_col)

    # 7. Feature Group Analysis
    feature_groups, group_stats = analyze_feature_groups(df_reduced, target_corr, pca_importance, mi_scores)

    # 8. Calculate Composite Scores
    sorted_scores = calculate_composite_scores(df_reduced, target_corr, pca_importance, cv_df, mi_scores)

    # 9. Final Feature Selection
    selected_features = select_best_features(df_reduced, sorted_scores, feature_groups, n_total=n_features)

    return {
        'selected_features': selected_features,
        'scores': sorted_scores,
        'groups': feature_groups,
        'group_stats': group_stats,
        'correlations': target_corr,
        'high_correlations': high_corr_pairs,
        'stability': cv_df,
        'pca_importance': pca_importance,
        'pca': pca,
        'explained_variance': explained_variance,
        'mi_scores': mi_scores,
        'removed_features': removed_features
    }


if __name__ == "__main__":
    try:
        file_path = '/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_us_historical_data/btcusdc_15m_historical.csv'
        # Load data
        df = load_data(file_path)

        # Run feature selection pipeline
        results = feature_selection_pipeline(df)

        # Access selected features
        print(f"\nBest features for prediction: {results['selected_features']}")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")