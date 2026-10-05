"""
models/baselines.py — USGS Fixed Logistic Regression baseline.

The USGS model is never trained or tuned — coefficients are fixed from
Staley et al. (2016), USGS Open-File Report 2016-1106.

Formula:  P = sigmoid(x),  x = -3.63 + 0.41·X1·R + 0.67·X2·R + 0.70·X3·R
  X1 = PropHM23  (proportion high/mod burn severity & slope ≥ 23°)
  X2 = dNBR/1000 (normalised burn ratio)
  X3 = KF-factor (soil erodibility)
  R  = Acc015_mm (15-min peak rainfall accumulation, mm)
"""

USGS_INTERCEPT = -3.63
USGS_COEFF_X1  =  0.41
USGS_COEFF_X2  =  0.67
USGS_COEFF_X3  =  0.70


def usgs_logit(X1, X2, X3, R) -> float:
    return (USGS_INTERCEPT
            + USGS_COEFF_X1 * X1 * R
            + USGS_COEFF_X2 * X2 * R
            + USGS_COEFF_X3 * X3 * R)


def usgs_probability(X1, X2, X3, R) -> float:
    import math
    x = usgs_logit(X1, X2, X3, R)
    ex = math.exp(x)
    return ex / (1 + ex)


def usgs_threat_score(df, threshold: float = 0.5) -> float:
    """
    Compute USGS-LR Threat Score on a DataFrame that has columns
    'usgs_p' (predicted probability) and 'pfdf' (true label).
    """
    y_pred = (df["usgs_p"].values >= threshold).astype(int)
    y_true = df["pfdf"].values.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")
