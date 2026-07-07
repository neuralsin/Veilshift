import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from science.thermal.module_b import generate_thermal_frame

def test_thermal_generator_class_invariance():
    """
    CLASS-CONDITIONAL THERMAL BACKGROUND/NOISE ARTIFACT regression test.
    Ensures that generating a target vs clutter sample does not alter the global noise distribution.
    """
    rng = np.random.default_rng(101)
    
    bg_T = 290.0
    bg_eps = 0.98
    target_T = 310.0
    target_eps = 0.90
    netd_K = 0.05
    frame_size = (64, 64)
    noise_model = "white" # Use white noise for easier statistical comparison
    noise_alpha = 1.0
    
    # Generate multiple noise fields for H0 (clutter) and H1 (target)
    n_samples = 50
    h0_noise_vars = []
    h1_noise_vars = []
    
    for _ in range(n_samples):
        # H0 (target properties match background)
        _, _, n0 = generate_thermal_frame(bg_T, bg_eps, bg_T, bg_eps, netd_K, frame_size, noise_model, noise_alpha, 8, rng)
        h0_noise_vars.append(np.var(n0))
        
        # H1 (target properties active)
        _, _, n1 = generate_thermal_frame(target_T, target_eps, bg_T, bg_eps, netd_K, frame_size, noise_model, noise_alpha, 8, rng)
        h1_noise_vars.append(np.var(n1))
        
    mean_var_h0 = np.mean(h0_noise_vars)
    mean_var_h1 = np.mean(h1_noise_vars)
    
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(h0_noise_vars, h1_noise_vars)
    
    # If p_val < 0.01, there is a statistically significant difference
    assert p_val > 0.01, f"CLASS-CONDITIONAL THERMAL BACKGROUND/NOISE ARTIFACT detected! p-value: {p_val}"
    print(f"PASS: Thermal background/noise distribution is invariant to target class presence (p={p_val:.4f}).")

if __name__ == "__main__":
    test_thermal_generator_class_invariance()
