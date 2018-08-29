# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

from PyQt4.QtGui import (
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    QMainWindow as _QMainWindow,
    )
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.gui.supplywidget import SupplyWidget as _SupplyWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.gui.databasewidget import DatabaseWidget \
    as _DatabaseWidget
from hallbench.gui.voltageoffsetwidget import VoltageOffsetWidget \
    as _VoltageOffsetWidget
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

        self.voltageoffset_tab = _VoltageOffsetWidget(self)
        self.ui.main_tab.addTab(self.voltageoffset_tab, 'Voltage Offset')

        self.temperature_tab = _TemperatureWidget(self)
        self.ui.main_tab.addTab(self.temperature_tab, 'Temperature')

        self.angularerror_tab = _AngularErrorWidget(self)
        self.ui.main_tab.addTab(self.angularerror_tab, 'Angular Error')

        self.database_tab = _DatabaseWidget(self)
        self.ui.main_tab.addTab(self.database_tab, 'Database')

        self.ui.database_le.setText(self.database)

        self.updateMainTabStatus()
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
            event.accept()
        except Exception:
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

    def updateMainTabStatus(self):
        """Enable or disable main tabs."""
        pass
