# -*- coding: utf-8 -*-

"""Main entry poin to the monitor application."""

import os as _os
import sys as _sys
import threading as _threading
from qtpy.QtWidgets import QApplication as _QApplication

from hallbench.gui import utils as _utils
from hallbench.gui.monitorwindow import MonitorWindow as _Window


WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 900


class App(_QApplication):
    """Application."""

    def __init__(self, args):
        """Start application."""
        super().__init__(args)
        self.setStyle(_utils.WINDOW_STYLE)

        self.directory = _utils.BASEPATH
        self.database_name = _utils.DATABASE_NAME
        self.mongo = _utils.MONGO
        self.server = _utils.SERVER


class GUIThread(_threading.Thread):
    """GUI Thread."""

    def __init__(self):
        """Start thread."""
        _threading.Thread.__init__(self)
        self.app = None
        self.window = None
        self.daemon = True
        self.start()

    def run(self):
        """Thread target function."""
        self.app = None
        if not _QApplication.instance():
            self.app = App([])
            self.window = _Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
            self.window.show()
            self.window.centralize_window()
            _sys.exit(self.app.exec_())


def run():
    """Run hallbench application."""
    app = None
    if not _QApplication.instance():
        app = App([])
        window = _Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        window.show()
        window.centralize_window()
        _sys.exit(app.exec_())


def run_in_thread():
    """Run hallbench application in a thread."""
    return GUIThread()
