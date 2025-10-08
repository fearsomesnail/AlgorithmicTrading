"""Determinism utilities for reproducible experiments."""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Seeds set to {seed} for reproducibility")
    logger.info(f"Deterministic algorithms enabled: {torch.are_deterministic_algorithms_enabled()}")


def test_reproducibility(func, *args, **kwargs):
    """Test that a function produces identical results with same seed."""
    # Run twice with same seed
    set_seeds(42)
    result1 = func(*args, **kwargs)
    
    set_seeds(42)
    result2 = func(*args, **kwargs)
    
    # Compare results
    if isinstance(result1, (int, float)):
        assert abs(result1 - result2) < 1e-6, f"Results differ: {result1} vs {result2}"
    elif isinstance(result1, np.ndarray):
        assert np.allclose(result1, result2), f"Arrays differ: max diff {np.max(np.abs(result1 - result2))}"
    else:
        assert result1 == result2, f"Results differ: {result1} vs {result2}"
    
    logger.info("Reproducibility test passed")
    return result1
