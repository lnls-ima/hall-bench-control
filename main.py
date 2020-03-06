# -*- coding: utf-8 -*-

"""Run the Hall bench control application."""

from hallbench.gui import hallbenchapp


THREAD = False


if THREAD:
    thread = hallbenchapp.run_in_thread()
else:
    hallbenchapp.run()


