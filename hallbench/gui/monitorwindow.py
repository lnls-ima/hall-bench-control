# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import sys as _sys
import traceback as _traceback
from qtpy.QtWidgets import (
    QFileDialog as _QFileDialog,
    QMainWindow as _QMainWindow,
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    )
from qtpy.QtCore import QTimer as _QTimer
import qtpy.uic as _uic

from hallbench.gui import utils as _utils
from hallbench.gui.auxiliarywidgets import (
    LogDialog as _LogDialog
    )
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.monitorpowersupplywidget import MonitorPowerSupplyWidget \
    as _PowerSupplyWidget
from hallbench.gui.currentwidget import CurrentWidget \
    as _CurrentWidget
from hallbench.gui.databasewidget import DatabaseWidget \
    as _DatabaseWidget
from hallbench.devices import logfile as _logfile


class MonitorWindow(_QMainWindow):
    """Main Window class for the monitor application."""

    _update_positions_interval = _utils.UPDATE_POSITIONS_INTERVAL

    def __init__(
            self, parent=None, width=_utils.WINDOW_WIDTH,
            height=_utils.WINDOW_HEIGHT):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)
        self.resize(width, height)

        # clear the current tabs
        self.ui.twg_main.clear()

        # define tab names and corresponding widgets
        self.tab_names = [
            'connection',
            'power_supply',
            'current',
            'database'
            ]

        self.tab_widgets = [
            _ConnectionWidget,
            _PowerSupplyWidget,
            _CurrentWidget,
            _DatabaseWidget,
            ]

        for idx, tab_name in enumerate(self.tab_names):
            tab_attr = 'tab_' + tab_name
            tab_name_split = tab_name.split('_')
            tab_label = ' '.join([s.capitalize() for s in tab_name_split])
            tab = self.tab_widgets[idx]()
            setattr(self, tab_attr, tab)
            self.ui.twg_main.addTab(tab, tab_label)

        self.log_dialog = _LogDialog()

        # show database name
        self.ui.le_database.setText(self.database_name)

        # connect signals and slots
        self.connect_signal_slots()

    @property
    def database_name(self):
        """Return the database name."""
        return _QApplication.instance().database_name

    @database_name.setter
    def database_name(self, value):
        _QApplication.instance().database_name = value

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            for idx in range(self.ui.twg_main.count()):
                widget = self.ui.twg_main.widget(idx)
                widget.close()
            self.log_dialog.close()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def change_database(self):
        """Change database file."""
        fn = _QFileDialog.getOpenFileName(
            self, caption='Database file', directory=self.directory,
            filter="Database File (*.db)")

        if isinstance(fn, tuple):
            fn = fn[0]

        if len(fn) == 0:
            return

        self.database_name = fn
        self.ui.le_database.setText(self.database_name)

    def centralize_window(self):
        """Centralize window."""
        window_center = _QDesktopWidget().availableGeometry().center()
        self.move(
            window_center.x() - self.geometry().width()/2,
            window_center.y() - self.geometry().height()/2)

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.tbt_database.clicked.connect(self.change_database)
        self.ui.tbt_log.clicked.connect(self.open_log)

    def open_log(self):
        """Open log info."""
        try:
            with open(_logfile, 'r') as f:
                text = f.read()
            self.log_dialog.te_text.setText(text)
            vbar = self.log_dialog.te_text.verticalScrollBar()
            vbar.setValue(vbar.maximum())
            self.log_dialog.show()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)