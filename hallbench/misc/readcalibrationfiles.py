# -*- coding: utf-8 -*-

import os as _os
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt

from hallbench.data import calibration as _calibration


d = (
    "C:\\Arq\\Work_At_LNLS\\eclipse-workspace\\" +
    "hall-bench-control\\fieldmaps\\CalibrationData\\FilesToDatabase\\")

max_order = 20

filenames = [
    f for f in _os.listdir(d) if f.endswith('txt')]

ns = []
for f in filenames:
    hc = _calibration.HallCalibrationCurve()
    hc.read_file(_os.path.join(d, f))
    ns.append(hc.calibration_name)

df = _pd.DataFrame(
    _np.zeros([len(filenames), max_order]),
    columns=range(max_order),
    index=ns)

for f in filenames:
    hc = _calibration.HallCalibrationCurve()
    hc.read_file(_os.path.join(d, f))
    coeffs = hc.polynomial_coeffs
    for idx, c in enumerate(coeffs):
        df.at[hc.calibration_name, idx] = c

df.to_clipboard(excel=True)
print(df)
