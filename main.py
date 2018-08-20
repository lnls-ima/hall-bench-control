# -*- coding: utf-8 -*-

"""Run the Hall bench control application."""

from hallbench import hallbenchapp


_thread = True


if (__name__ == '__main__'):
    if _thread:
        thread = hallbenchapp.run_in_thread()
    else:
        hallbenchapp.run()