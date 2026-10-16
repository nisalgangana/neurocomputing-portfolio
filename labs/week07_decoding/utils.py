import numpy as np

def wrap_angle_deg(angle):
    return angle % 360

def angular_error_deg(decoded, true):
    return ((decoded - true + 180) % 360) - 180

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot