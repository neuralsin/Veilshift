import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from science.fusion.module_h import select_threshold_neyman_pearson

def test_threshold_selection_bug():
    """
    PERFECTLY SEPARABLE SCORES regression test.
    Ensures that when a finite operating point exists (Pd=1, FAR=0), it is selected
    over a Sentinel Infinity threshold that yields Pd=0.
    """
    labels = np.array([0, 0, 0, 1, 1, 1])
    fused_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    threshold, pd, far = select_threshold_neyman_pearson(fused_scores, labels, target_far=0.01)
    
    # We expect a finite threshold between 0.3 and 0.7 that yields Pd=1, FAR=0
    # In sklearn's roc_curve, the thresholds for perfect separation usually return 
    # the lowest positive score or a point between the classes.
    # We want to ensure the threshold is finite (not inf) and Pd=1.0.
    
    assert np.isfinite(threshold), f"ROC INF SENTINEL / THRESHOLD SELECTION BUG detected! Threshold: {threshold}"
    assert pd == 1.0, f"Expected Pd=1.0 for trivially separable data, got {pd}"
    assert far == 0.0, f"Expected FAR=0.0, got {far}"
    print(f"PASS: Valid finite threshold selected. Threshold={threshold}, Pd={pd}, FAR={far}")

if __name__ == "__main__":
    test_threshold_selection_bug()
