# -*- coding: utf-8 -*-

"""Main entry poin to the Hall bench control application."""

import os as _os
import sys as _sys
import threading as _threading
from qtpy.QtWidgets import QApplication as _QApplication

from hallbench.calibration.calibrationwindow import (
    CalibrationWindow as _CalibrationWindow)
import hallbench.devices as _devices


# Style: ["windows", "motif", "cde", "plastique", "windowsxp", or "macintosh"]
_style = 'windows'
_width = 1200
_height = 700


class CalibrationApp(_QApplication):
    """Hall bench application."""

    def __init__(self, args):
        """Start application."""
        super().__init__(args)
        self.setStyle(_style)
        self.nmr = _devices.NMRLib.NMR()
        self.mult = _devices.GPIBLib.Agilent3458A()
        self.mch = _devices.SerialLib.Agilent34970A()
       

class GUIThread(_threading.Thread):
    """GUI Thread."""

    def __init__(self):
        """Start thread."""
        _threading.Thread.__init__(self)
        self.app = None
        self.window = None
        self.start()

    def run(self):
        """Thread target function."""
        self.app = None
        if (not _QApplication.instance()):
            self.app = CalibrationApp([])
            self.window = _CalibrationWindow(width=_width, height=_height)
            self.window.show()
            self.window.centralizeWindow()
            _sys.exit(self.app.exec_())


def run():
    """Run hallbench application."""
    app = None
    if (not _QApplication.instance()):
        app = CalibrationApp([])
        window = _CalibrationWindow(width=_width, height=_height)
        window.show()
        window.centralizeWindow()
        _sys.exit(app.exec_())


def run_in_thread():
    """Run hallbench application in a thread."""
    return GUIThread()
