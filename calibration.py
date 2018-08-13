# -*- coding: utf-8 -*-

"""Run the Hall probe calibration application."""

from hallbench import calibrationapp


_thread = True


if (__name__ == '__main__'):
    if _thread:
        thread = calibrationapp.run_in_thread()
    else:
        calibrationapp.run()
