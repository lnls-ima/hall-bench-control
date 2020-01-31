# -*- coding: utf-8 -*-

"""Run the Hall bench control application."""

from hallbench.gui import hallbenchapp


_thread = True


if _thread:
    thread = hallbenchapp.run_in_thread()
else:
    hallbenchapp.run()
