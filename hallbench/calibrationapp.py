# -*- coding: utf-8 -*-

"""Main entry poin to the Hall probe calibration application."""

import sys as _sys
import threading as _threading
from PyQt5.QtWidgets import QApplication as _QApplication


_style = 'windows'


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
            self.app = _QApplication([])
            self.app.setStyle(_style)

            from hallbench.gui.calibrationwindow import CalibrationWindow
            self.window = CalibrationWindow()
            self.window.show()

            _sys.exit(self.app.exec_())


def run():
    """Run calibration application."""
    app = None
    if (not _QApplication.instance()):
        app = _QApplication([])
        app.setStyle(_style)

        from hallbench.gui.calibrationwindow import CalibrationWindow
        window = CalibrationWindow()
        window.show()

        _sys.exit(app.exec_())


def run_in_thread():
    """Run calibration application in a thread."""
    return GUIThread()
