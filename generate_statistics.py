"""
Generate statistical results for the UCI Heart Disease dataset.
Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy import stats


def load_heart_disease_data():
    """Fetch the Heart Disease dataset from UCI ML Repository."""
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    return df, X, y, heart_disease.variables


def generate_statistics(output_file="statistical_results.txt"):
    """Generate comprehensive statistical analysis of the Heart Disease dataset."""
    print("Loading Heart Disease dataset from UCI ML Repository...")
    df, X, y, variables = load_heart_disease_data()

    # Target column: typically 'num' or 'HeartDisease'
    target_col = y.columns[0] if len(y.columns) > 0 else df.columns[-1]

    results = []
    results.append("=" * 80)
    results.append("HEART DISEASE DATASET - STATISTICAL ANALYSIS")
    results.append("Source: UCI ML Repository - https://archive.ics.uci.edu/dataset/45/heart+disease")
    results.append("=" * 80)

    # 1. Dataset Overview
    results.append("\n" + "-" * 80)
    results.append("1. DATASET OVERVIEW")
    results.append("-" * 80)
    results.append(f"Number of instances: {len(df)}")
    results.append(f"Number of features: {len(X.columns)}")
    results.append(f"Target variable: {target_col}")
    results.append(f"\nColumn names: {list(df.columns)}")

    # 2. Data Types
    results.append("\n" + "-" * 80)
    results.append("2. DATA TYPES")
    results.append("-" * 80)
    results.append(df.dtypes.to_string())

    # 3. Missing Values
    results.append("\n" + "-" * 80)
    results.append("3. MISSING VALUES")
    results.append("-" * 80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        results.append(missing[missing > 0].to_string())
    else:
        results.append("No missing values detected.")
    # Check for placeholder missing values (e.g., -9, ?, etc.)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isin([-9, -1]).any():
            count = df[col].isin([-9, -1]).sum()
            if count > 0:
                results.append(f"\nNote: Column '{col}' contains {count} placeholder values (-9/-1) often used for missing.")

    # 4. Descriptive Statistics (Numeric)
    results.append("\n" + "-" * 80)
    results.append("4. DESCRIPTIVE STATISTICS (All Numeric Columns)")
    results.append("-" * 80)
    results.append(df.describe(include=[np.number]).to_string())

    # 5. Target Variable Distribution
    results.append("\n" + "-" * 80)
    results.append("5. TARGET VARIABLE DISTRIBUTION")
    results.append("-" * 80)
    target_counts = df[target_col].value_counts().sort_index()
    results.append(target_counts.to_string())
    results.append(f"\nBinary interpretation (0=no disease, 1-4=presence):")
    no_disease = (df[target_col] == 0).sum()
    has_disease = (df[target_col] > 0).sum()
    results.append(f"  No heart disease (0): {no_disease} ({100*no_disease/len(df):.1f}%)")
    results.append(f"  Heart disease (1-4): {has_disease} ({100*has_disease/len(df):.1f}%)")

    # 6. Categorical/Cardinal Feature Value Counts
    results.append("\n" + "-" * 80)
    results.append("6. CATEGORICAL FEATURE VALUE COUNTS")
    results.append("-" * 80)
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_cols:
        if col in df.columns:
            results.append(f"\n{col}:")
            results.append(df[col].value_counts().sort_index().to_string())

    # 7. Correlation Matrix (Numeric features)
    results.append("\n" + "-" * 80)
    results.append("7. CORRELATION MATRIX (Numeric Features)")
    results.append("-" * 80)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    results.append(corr_matrix.round(3).to_string())

    # 8. Key Correlations with Target
    results.append("\n" + "-" * 80)
    results.append("8. FEATURE CORRELATIONS WITH TARGET")
    results.append("-" * 80)
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col, errors='ignore')
        target_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
        results.append(target_corr.round(4).to_string())

    # 9. Normality Tests (Shapiro-Wilk for key numeric features)
    results.append("\n" + "-" * 80)
    results.append("9. NORMALITY TEST (Shapiro-Wilk) - Key Continuous Features")
    results.append("-" * 80)
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in continuous_cols:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 3 and len(data) < 5000:
                stat, p_value = stats.shapiro(data)
                results.append(f"{col}: statistic={stat:.4f}, p-value={p_value:.4f} {'(normal)' if p_value > 0.05 else '(non-normal)'}")

    # 10. Skewness and Kurtosis
    results.append("\n" + "-" * 80)
    results.append("10. SKEWNESS AND KURTOSIS")
    results.append("-" * 80)
    skew_kurt = pd.DataFrame({
        'skewness': numeric_df.skew(),
        'kurtosis': numeric_df.kurtosis()
    }).round(4)
    results.append(skew_kurt.to_string())

    # Write to file
    output_path = output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

    print(f"\nStatistical results written to '{output_path}'")
    return '\n'.join(results)


if __name__ == "__main__":
    results = generate_statistics()
    print("\n" + results)
