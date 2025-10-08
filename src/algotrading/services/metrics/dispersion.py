"""Cross-sectional dispersion metrics for model evaluation."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def cross_sectional_dispersion(pred: np.ndarray, target: np.ndarray, dates: np.ndarray) -> tuple:
    """
    Calculate cross-sectional dispersion ratio across dates.
    
    Args:
        pred: Predictions array [N]
        target: Target values array [N] 
        dates: Date array [N] (timestamp or int day id)
    
    Returns:
        (ratio_mean, pred_cs_mean, tgt_cs_mean): Average ratio and mean dispersions
    """
    df = pd.DataFrame({"pred": pred, "tgt": target, "d": dates})
    g = df.groupby("d")
    
    # Calculate cross-sectional standard deviation per date
    pred_cs = g["pred"].std(ddof=0)
    tgt_cs = g["tgt"].std(ddof=0)
    
    # Calculate ratio, handling zero target std
    ratio_series = pred_cs / (tgt_cs.replace(0, np.nan))
    
    # Return mean ratio and mean dispersions
    ratio_mean = float(ratio_series.mean(skipna=True))
    pred_cs_mean = float(pred_cs.mean())
    tgt_cs_mean = float(tgt_cs.mean())
    
    logger.info(f"Cross-sectional dispersion: ratio={ratio_mean:.3f}, pred_cs={pred_cs_mean:.6f}, tgt_cs={tgt_cs_mean:.6f}")
    
    return ratio_mean, pred_cs_mean, tgt_cs_mean


def check_prediction_collapse(pred: np.ndarray, target: np.ndarray, dates: np.ndarray, 
                            threshold: float = 0.25) -> tuple:
    """
    Check for prediction collapse using cross-sectional dispersion.
    
    Args:
        pred: Predictions array
        target: Target values array
        dates: Date array
        threshold: Minimum ratio threshold (default 0.25)
    
    Returns:
        (is_collapsed, ratio, pred_cs, tgt_cs): Collapse status and metrics
    """
    ratio, pred_cs, tgt_cs = cross_sectional_dispersion(pred, target, dates)
    is_collapsed = ratio < threshold
    
    if is_collapsed:
        logger.warning(f"PREDICTION COLLAPSE: Cross-sectional dispersion ratio {ratio:.3f} < {threshold}")
    else:
        logger.info(f"Prediction dispersion OK: ratio {ratio:.3f} >= {threshold}")
    
    return is_collapsed, ratio, pred_cs, tgt_cs
