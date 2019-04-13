# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import sys as _sys
import traceback as _traceback
from qtpy.QtWidgets import (
    QDialog as _QDialog,    
    QCheckBox as _QCheckBox,
    QGroupBox as _QGroupBox,
    QPushButton as _QPushButton,
    QFileDialog as _QFileDialog,
    QMainWindow as _QMainWindow,
    QVBoxLayout as _QVBoxLayout,
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    )
from qtpy.QtGui import (
    QFont as _QFont,
    )
from qtpy.QtCore import (
    QSize as _QSize,
    QTimer as _QTimer,
    Signal as _Signal,
    QRunnable as _QRunnable,
    QThreadPool as _QThreadPool,
    )
import qtpy.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.gui.supplywidget import SupplyWidget as _SupplyWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.gui.voltagewidget import VoltageWidget \
    as _VoltageWidget
from hallbench.gui.pscurrentwidget import PSCurrentWidget \
    as _PSCurrentWidget
from hallbench.gui.temperaturewidget import TemperatureWidget \
    as _TemperatureWidget
from hallbench.gui.coolingsystemwidget import CoolingSystemWidget \
    as _CoolingSystemWidget
from hallbench.gui.angularerrorwidget import AngularErrorWidget \
    as _AngularErrorWidget
from hallbench.gui.voltagetemperaturewidget import VoltageTempWidget \
    as _VoltageTempWidget
from hallbench.gui.databasewidget import DatabaseWidget \
    as _DatabaseWidget


class HallBenchWindow(_QMainWindow):
    """Main Window class for the Hall Bench Control application."""

    _update_positions_timer_interval = 500  # [ms]

    def __init__(self, parent=None, width=1200, height=700):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)
        self.resize(width, height)

        # clear the current tabs
        self.ui.main_tab.clear()

        # define tab names and corresponding widgets
        self.tab_names = [
            'connection',
            'motors',
            'power_supply',
            'measurement',
            'voltage',
            'current',
            'temperature',
            'cooling_system',
            'angular_error',
            'voltage_temperature',
            'database',
            ]
        
        self.tab_widgets = [
            _ConnectionWidget,
            _MotorsWidget,
            _SupplyWidget,
            _MeasurementWidget,
            _VoltageWidget,
            _PSCurrentWidget,
            _TemperatureWidget,
            _CoolingSystemWidget,
            _AngularErrorWidget,
            _VoltageTempWidget,
            _DatabaseWidget,
            ]

        # add preferences dialog
        self.preferences_dialog = PreferencesDialog(self.tab_names)
        self.preferences_dialog.preferences_changed.connect(self.changeTabs)

        # show database name
        self.ui.database_le.setText(self.database)
        
        # start positions update
        self.stop_positions_update = False
        self.threadpool = _QThreadPool.globalInstance()
        self.timer = _QTimer()
        self.timer.timeout.connect(self.updatePositions)
        self.timer.start(self._update_positions_timer_interval)

        # connect signals and slots 
        self.connectSignalSlots()

    @property
    def database(self):
        """Return the database filename."""
        return _QApplication.instance().database

    @database.setter
    def database(self, value):
        _QApplication.instance().database = value

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    @property
    def save_fieldmap_dialog(self):
        """Save fieldmap dialog."""
        return _QApplication.instance().save_fieldmap_dialog

    @property
    def view_probe_dialog(self):
        """View probe dialog."""
        return _QApplication.instance().view_probe_dialog

    @property
    def view_scan_dialog(self):
        """View scan dialog."""
        return _QApplication.instance().view_scan_dialog

    @property
    def view_fieldmap_dialog(self):
        """View fieldmap dialog."""
        return _QApplication.instance().view_fieldmap_dialog

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            self.stop_positions_update = True
            self.threadpool.waitForDone()
            self.timer.stop()
            for idx in range(self.ui.main_tab.count()):
                widget = self.ui.main_tab.widget(idx)
                widget.close()
            self.view_probe_dialog.accept()
            self.view_scan_dialog.accept()
            self.save_fieldmap_dialog.accept()
            self.view_fieldmap_dialog.accept()
            self.preferences_dialog.close()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def changeDatabase(self):
        """Change database file."""
        fn = _QFileDialog.getOpenFileName(
            self, caption='Database file', directory=self.directory,
            filter="Database File (*.db)")

        if isinstance(fn, tuple):
            fn = fn[0]

        if len(fn) == 0:
            return

        self.database = fn
        self.ui.database_le.setText(self.database)

    def changeTabs(self, tab_status):
        """Hide or show tabs."""
        try:
            if self.ui.main_tab.count() != 0:
                current_tab = self.ui.main_tab.currentWidget()
            else:
                current_tab = None
            
            self.ui.main_tab.clear()
            for idx, tab_name in enumerate(self.tab_names):
                tab_attr = tab_name + '_tab'
                tab_label = tab_name.replace('_', ' ').capitalize()
                status = tab_status[tab_name]
                if status:
                    if hasattr(self, tab_attr):
                        tab = getattr(self, tab_attr)
                    else:
                        tab = self.tab_widgets[idx]()
                        setattr(self, tab_attr, tab)
                    self.ui.main_tab.addTab(tab, tab_label)

            if current_tab is not None:
                idx = self.ui.main_tab.indexOf(current_tab)
                self.ui.main_tab.setCurrentIndex(idx)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def centralizeWindow(self):
        """Centralize window."""
        window_center = _QDesktopWidget().availableGeometry().center()
        self.move(
            window_center.x() - self.geometry().width()/2,
            window_center.y() - self.geometry().height()/2)

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        if (hasattr(self, 'measurement_tab') and 
            hasattr(self, 'power_supply_tab')):
            self.measurement_tab.change_current_setpoint.connect(
                self.power_supply_tab.change_setpoint_and_emit_signal)
            
            self.measurement_tab.turn_off_power_supply_current.connect(
                self.power_supply_tab.set_current_to_zero)
            
            self.power_supply_tab.current_setpoint_changed.connect(
                self.measurement_tab.updateCurrentSetpoint)
            
            self.power_supply_tab.start_measurement.connect(
                self.measurement_tab.measureAndEmitSignal)
            
            self.power_supply_tab.current_ramp_end.connect(
                self.measurement_tab.endAutomaticMeasurements)
        
        self.preferences_dialog.tabsPreferencesChanged()
        self.ui.preferences_btn.clicked.connect(self.preferences_dialog.show)
        self.ui.database_btn.clicked.connect(self.changeDatabase)

    def updatePositions(self):
        """Update pmac positions."""
        if self.stop_positions_update:
            return
 
        try:
            worker = PositionsWorker()
            self.threadpool.start(worker)
        except Exception:
            pass


class PreferencesDialog(_QDialog):
    """Preferences dialog class for Hall Bench Control application."""

    preferences_changed = _Signal([dict])

    def __init__(self, chb_names, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(250, 400)

        font = _QFont()
        font.setPointSize(11)
        font.setBold(False)
        self.setFont(font)

        font_bold = _QFont()
        font_bold.setPointSize(11)
        font_bold.setBold(True)
    
        main_layout = _QVBoxLayout()
        vertical_layout = _QVBoxLayout()        
        group_box = _QGroupBox("Select Tabs to Show")
        group_box.setLayout(vertical_layout)
        group_box.setFont(font_bold)
        main_layout.addWidget(group_box)
        self.setLayout(main_layout)
        
        self.chb_names = chb_names
        for name in self.chb_names:
            label = name.replace('_', ' ').capitalize()
            chb = _QCheckBox(label)
            setattr(self, name + '_chb', chb)
            vertical_layout.addWidget(chb)
            chb.setFont(font)
        
        self.apply_btn = _QPushButton("Apply Changes")
        self.apply_btn.setMinimumSize(_QSize(0, 40))
        self.apply_btn.setFont(font_bold)
        vertical_layout.addWidget(self.apply_btn)         

        self.apply_btn.clicked.connect(self.tabsPreferencesChanged)
        self.connection_chb.setChecked(True)
        self.motors_chb.setChecked(True)
        self.measurement_chb.setChecked(True)

    def tabsPreferencesChanged(self):
        """Get tabs checkbox status and emit signal to change tabs."""
        try:
            chb_status = {}
            for chb_name in self.chb_names:
                chb = getattr(self, chb_name + '_chb')
                chb_status[chb_name] = chb.isChecked()

            self.preferences_changed.emit(chb_status)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class PositionsWorker(_QRunnable):
    """Read position values from pmac."""

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
 
    def run(self):
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
