# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import sys as _sys
import traceback as _traceback
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    QMainWindow as _QMainWindow,
    QDialog as _QDialog,
    )
from PyQt5.QtCore import (
    QTimer as _QTimer,
    QThread as _QThread,
    QEventLoop as _QEventLoop,
    pyqtSignal as _pyqtSignal,
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
from hallbench.gui.pscurrentwidget import PSCurrentWidget \
    as _PSCurrentWidget
from hallbench.gui.temperaturewidget import TemperatureWidget \
    as _TemperatureWidget
from hallbench.gui.angularerrorwidget import AngularErrorWidget \
    as _AngularErrorWidget
from hallbench.gui.coolingsystemwidget import CoolingSystemWidget \
    as _CoolingSystemWidget


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
        self.changing_tabs = False

        # add preferences dialog
        self.preferences_dialog = PreferencesDialog()
        self.preferences_dialog.preferences_changed.connect(self.changeTabs)

        # add tabs
        self.connection_tab = _ConnectionWidget(self)
        self.ui.main_tab.addTab(self.connection_tab, 'Connection')

        self.motors_tab = _MotorsWidget(self)
        self.ui.main_tab.addTab(self.motors_tab, 'Motors')

        self.power_supply_tab = _SupplyWidget(self)
        self.ui.main_tab.addTab(self.power_supply_tab, 'Power Supply')

        self.measurement_tab = _MeasurementWidget(self)
        self.ui.main_tab.addTab(self.measurement_tab, 'Measurement')

        self.voltage_tab = _VoltageWidget(self)
        self.ui.main_tab.addTab(self.voltage_tab, 'Voltage')

        self.current_tab = _PSCurrentWidget(self)
        self.ui.main_tab.addTab(self.current_tab, 'Current')

        self.temperature_tab = _TemperatureWidget(self)
        self.ui.main_tab.addTab(self.temperature_tab, 'Temperature')

        self.cooling_system_tab = _CoolingSystemWidget(self)
        self.ui.main_tab.addTab(self.cooling_system_tab, 'Cooling System')

        self.angular_error_tab = _AngularErrorWidget(self)
        self.ui.main_tab.addTab(self.angular_error_tab, 'Angular Error')

        self.database_tab = _DatabaseWidget(self)
        self.ui.main_tab.addTab(self.database_tab, 'Database')

        self.ui.database_le.setText(self.database)

        self.positions_thread = PositionsThread()
        self.positions_thread.start()

        # Connect automatic current ramp signals
        self.measurement_tab.change_current_setpoint.connect(
            self.power_supply_tab.change_setpoint_and_emit_signal)

        self.measurement_tab.turn_off_power_supply.connect(
            self.power_supply_tab.turn_off)

        self.power_supply_tab.current_setpoint_changed.connect(
            self.measurement_tab.updateCurrentSetpoint)

        self.power_supply_tab.start_measurement.connect(
            self.measurement_tab.measureAndEmitSignal)

        self.power_supply_tab.current_ramp_end.connect(
            self.measurement_tab.endAutomaticMeasurements)

        self.ui.preferences_btn.clicked.connect(self.preferences_dialog.show)
        self.preferences_dialog.tabsPreferencesChanged()

        self.ui.main_tab.currentChanged.connect(self.updateDatabaseTab)

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
            self.preferences_dialog.close()
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

    def changeTabs(self, tab_status):
        """Hide or show tabs."""
        try:
            self.changing_tabs = True
            current_tab = self.ui.main_tab.currentWidget()
            self.ui.main_tab.clear()
            sorted_ts = sorted(tab_status.items(), key=lambda x: x[1][1])
            for i in range(len(sorted_ts)):
                tab_name = sorted_ts[i][0]
                status = sorted_ts[i][1][0]
                if status and hasattr(self, tab_name + '_tab'):
                    tab = getattr(self, tab_name + '_tab')
                    tab_label = tab_name.replace('_', ' ').capitalize()
                    self.ui.main_tab.addTab(tab, tab_label)

            idx = self.ui.main_tab.indexOf(current_tab)
            self.ui.main_tab.setCurrentIndex(idx)
            self.changing_tabs = False

        except Exception:
            self.changing_tabs = False
            _traceback.print_exc(file=_sys.stdout)

    def updateDatabaseTab(self):
        """Update database tab."""
        if self.changing_tabs:
            return

        database_tab_idx = self.ui.main_tab.indexOf(self.database_tab)
        if self.ui.main_tab.currentIndex() == database_tab_idx:
            self.database_tab.updateDatabaseTables()


class PreferencesDialog(_QDialog):
    """Preferences dialog class for Hall Bench Control application."""

    preferences_changed = _pyqtSignal([dict])

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)
        self.ui.apply_btn.clicked.connect(self.tabsPreferencesChanged)
        self.ui.connection_chb.setChecked(True)
        self.ui.motors_chb.setChecked(True)
        self.ui.power_supply_chb.setChecked(True)
        self.ui.measurement_chb.setChecked(True)
        self.ui.voltage_chb.setChecked(False)
        self.ui.current_chb.setChecked(False)
        self.ui.temperature_chb.setChecked(False)
        self.ui.cooling_system_chb.setChecked(False)
        self.ui.angular_error_chb.setChecked(False)
        self.ui.database_chb.setChecked(True)

    def tabsPreferencesChanged(self):
        """Get tabs checkbox status and emit signal to change tabs."""
        try:
            tab_status = {}
            tab_names = [
                'connection',
                'motors',
                'power_supply',
                'measurement',
                'voltage',
                'current',
                'temperature',
                'cooling_system',
                'angular_error',
                'database',
                ]
            idx = 0
            for tab_name in tab_names:
                chb = getattr(self.ui, tab_name + '_chb')
                if chb.isChecked():
                    tab_status[tab_name] = (True, idx)
                else:
                    tab_status[tab_name] = (False, idx)
                idx = idx + 1
            self.preferences_changed.emit(tab_status)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)


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
