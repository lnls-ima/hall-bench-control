# -*- coding: utf-8 -*-

"""Run the Hall bench calibration application."""

from hallbench.calibration import calibrationapp


_thread = True


if _thread:
    thread = calibrationapp.run_in_thread()
else:
    calibrationapp.run()