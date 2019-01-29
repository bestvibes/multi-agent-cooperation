from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

def interpolate_parametric(t, x, y):
    t_new = np.linspace(0, len(t)-1, 300)

    x_spl = make_interp_spline(t, x, k=3)
    x_smooth = x_spl(t_new)

    y_spl = make_interp_spline(t, y, k=3)
    y_smooth = y_spl(t_new)

    return (t_new, x_smooth, y_smooth)
