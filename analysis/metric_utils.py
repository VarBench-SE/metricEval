import numpy as np
import pandas as pd
import scipy.stats as stats

def pearson_confidence_interval(df, confidence=0.95):
    """Compute Pearson correlation, confidence intervals, and p-values for all metric columns vs 'human score'."""
    human_scores = df["human_score"]
    results = {}
    if np.all(human_scores == human_scores.iloc[0]):
        return pd.DataFrame(results).T
    for metric in df.columns:
        if metric in ["human score", "id","config"]:  # Exclude non-metric columns
            continue
        x = df[metric]
        if len(x) < 3:  # Avoid errors for small groups
            continue
        if np.all(x == x.iloc[0]):  # Check if the column is constant
            continue  # Skip this metric
        r, p_value = stats.pearsonr(x, human_scores)
        n = len(x)

        if r == 1 or r == -1:
            results[metric] ={"metric": metric, "Pearson": r, "95% CI L": np.nan, "95% CI H": np.nan, "p-value": p_value}
            continue
        # Fisher's Z transformation
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)

        # Confidence interval in Z-space
        z_low, z_high = z - z_crit * se, z + z_crit * se

        # Convert back to r-space
        r_low, r_high = np.tanh(z_low), np.tanh(z_high)

        results[metric] = {"metric": metric,"Pearson": r, "95% CI L": r_low,"95% CI H": r_high, "p-value": p_value}
    
    return pd.DataFrame(results).T

def kendall_confidence_interval(df, confidence=0.95):
    """Compute Kendall's Tau correlation, confidence intervals, and p-values for all metric columns vs 'human score'."""
    human_scores = df["human_score"]
    results = {}
    if np.all(human_scores == human_scores.iloc[0]):
        return pd.DataFrame(results).T
    for metric in df.columns:
        if metric in ["human score", "id","config"]:  # Exclude non-metric columns
            continue
        x = df[metric]
        if len(x) < 3:  # Avoid errors for small groups
            continue
        if np.all(x == x.iloc[0]):  # Check if the column is constant
            continue  # Skip this metric
        tau, p_value = stats.kendalltau(x, human_scores)
        n = len(x)

        # Standard error approximation for Kendall's Tau
        se = np.sqrt((2 * (2 * n + 5)) / (9 * n * (n - 1)))
        z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)

        # Confidence interval
        tau_low, tau_high = tau - z_crit * se, tau + z_crit * se

        results[metric] = {"metric": metric,"Kendall's Tau": tau, "95% CI L": tau_low,"95% CI H": tau_high, "p-value": p_value}
    
    return pd.DataFrame(results).T
