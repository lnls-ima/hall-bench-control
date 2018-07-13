# -*- coding: utf-8 -*-

"""Main entry poin to the Hall bench control application."""

import sys as _sys
import threading as _threading
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    )
from PyQt5.QtGui import QFont as _QFont

# Style: ["windows", "motif", "cde", "plastique", "windowsxp", or "macintosh"]
_style = 'motif'
_fontsize = 10
_window_width = 1200
_window_height = 900


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

            font = _QFont()
            font.setPointSize(_fontsize)

            from hallbench.gui.hallbenchwindow import HallBenchWindow
            self.window = HallBenchWindow()
            self.window.setFont(font)
            self.window.resize(_window_width, _window_height)

            self.window.show()
            window_center = _QDesktopWidget().availableGeometry().center()
            self.window.move(
                window_center.x() - self.window.geometry().width()/2,
                window_center.y() - self.window.geometry().height()/2)

            _sys.exit(self.app.exec_())


def run():
    """Run hallbench application."""
    app = None
    if (not _QApplication.instance()):
        app = _QApplication([])
        app.setStyle(_style)

        font = _QFont()
        font.setPointSize(_fontsize)

        from hallbench.gui.hallbenchwindow import HallBenchWindow
        window = HallBenchWindow()
        window.setFont(font)
        window.resize(_window_width, _window_height)

        window.show()
        window_center = _QDesktopWidget().availableGeometry().center()
        window.move(
            window_center.x() - window.geometry().width()/2,
            window_center.y() - window.geometry().height()/2)

        _sys.exit(app.exec_())


def run_in_thread():
    """Run hallbench application in a thread."""
    return GUIThread()
