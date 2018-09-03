# -*- coding: utf-8 -*-

"""Run the Hall bench control application."""

from hallbench.gui import hallbenchapp


_thread = True
_daemon = False


if (__name__ == '__main__'):
    if _thread:
        thread = hallbenchapp.run_in_thread(_daemon)
    else:
        hallbenchapp.run()