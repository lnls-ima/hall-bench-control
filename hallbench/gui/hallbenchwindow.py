# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import sys as _sys
import traceback as _traceback
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    QMainWindow as _QMainWindow,
    )
from PyQt5.QtCore import (
    QTimer as _QTimer,
    QThread as _QThread,
    QEventLoop as _QEventLoop,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.gui.supplywidget import SupplyWidget as _SupplyWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.gui.databasewidget import DatabaseWidget \
    as _DatabaseWidget
from hallbench.gui.voltagewidget import VoltageWidget \
    as _VoltageWidget
from hallbench.gui.temperaturewidget import TemperatureWidget \
    as _TemperatureWidget
from hallbench.gui.angularerrorwidget import AngularErrorWidget \
    as _AngularErrorWidget


class HallBenchWindow(_QMainWindow):
    """Main Window class for the Hall Bench Control application."""

    def __init__(self, parent=None, width=1200, height=700):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)
        self.resize(width, height)

        # clear the current tabs
        self.ui.main_tab.clear()

        # add tabs
        self.connection_tab = _ConnectionWidget(self)
        self.ui.main_tab.addTab(self.connection_tab, 'Connection')

        self.motors_tab = _MotorsWidget(self)
        self.ui.main_tab.addTab(self.motors_tab, 'Motors')

        self.supply_tab = _SupplyWidget(self)
        self.ui.main_tab.addTab(self.supply_tab, 'Power Supply')

        self.measurement_tab = _MeasurementWidget(self)
        self.ui.main_tab.addTab(self.measurement_tab, 'Measurement')

        self.voltage_tab = _VoltageWidget(self)
        self.ui.main_tab.addTab(self.voltage_tab, 'Voltage')

        self.temperature_tab = _TemperatureWidget(self)
        self.ui.main_tab.addTab(self.temperature_tab, 'Temperature')

        self.angularerror_tab = _AngularErrorWidget(self)
        self.ui.main_tab.addTab(self.angularerror_tab, 'Angular Error')

        self.database_tab = _DatabaseWidget(self)
        self.ui.main_tab.addTab(self.database_tab, 'Database')

        self.ui.database_le.setText(self.database)

        self.ui.main_tab.currentChanged.connect(self.updateDatabaseTab)

        self.positions_thread = PositionsThread()
        self.positions_thread.start()

        # Connect automatic current ramp signals
        self.measurement_tab.change_current_setpoint.connect(
            self.supply_tab.change_setpoint_and_emit_signal)

        self.measurement_tab.turn_off_power_supply.connect(
            self.supply_tab.turn_off)

        self.supply_tab.current_setpoint_changed.connect(
            self.measurement_tab.measureAndEmitSignal)

        self.supply_tab.current_ramp_end.connect(
            self.measurement_tab.endAutomaticMeasurements)

    @property
    def database(self):
        """Return the database filename."""
        return _QApplication.instance().database

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            for idx in range(self.ui.main_tab.count()):
                widget = self.ui.main_tab.widget(idx)
                widget.close()
            self.positions_thread.quit()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def centralizeWindow(self):
        """Centralize window."""
        window_center = _QDesktopWidget().availableGeometry().center()
        self.move(
            window_center.x() - self.geometry().width()/2,
            window_center.y() - self.geometry().height()/2)

    def updateDatabaseTab(self):
        """Update database tab."""
        database_tab_idx = self.ui.main_tab.indexOf(self.database_tab)
        if self.ui.main_tab.currentIndex() == database_tab_idx:
            self.database_tab.updateDatabaseTables()


class PositionsThread(_QThread):
    """Thread to read position values from pmac."""

    _timer_interval = 250  # [ms]

    def __init__(self):
        """Initialize object."""
        super().__init__()
        self.timer = _QTimer()
        self.timer.moveToThread(self)
        self.timer.timeout.connect(self.updatePositions)

    @property
    def pmac(self):
        """Pmac communication class."""
        return _QApplication.instance().devices.pmac

    @property
    def positions(self):
        """Get current posiitons dict."""
        return _QApplication.instance().positions

    @positions.setter
    def positions(self, value):
        _QApplication.instance().positions = value

    def updatePositions(self):
        """Update axes positions."""
        if not self.pmac.connected:
            self.positions = {}
            return

        try:
            for axis in self.pmac.commands.list_of_axis:
                pos = self.pmac.get_position(axis)
                self.positions[axis] = pos
        except Exception:
            pass

    def run(self):
        """Target function."""
        self.timer.start(self._timer_interval)
        loop = _QEventLoop()
        loop.exec_()
