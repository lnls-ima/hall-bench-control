# -*- coding: utf-8 -*-

import sys as _sys
import numpy as _np
import pandas as _pd
import datetime as _datetime
import warnings as _warnings
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QDialog as _QDialog,
    QLabel as _QLabel,
    QSpinBox as _QSpinBox,
    QGroupBox as _QGroupBox,
    QComboBox as _QComboBox,
    QCheckBox as _QCheckBox,
    QListView as _QListView,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    QSizePolicy as _QSizePolicy,
    QSpacerItem as _QSpacerItem,
    QPushButton as _QPushButton,
    QToolButton as _QToolButton,
    QHBoxLayout as _QHBoxLayout,
    QVBoxLayout as _QVBoxLayout,
    QGridLayout as _QGridLayout,
    QFormLayout as _QFormLayout,
    QFileDialog as _QFileDialog,
    QTableWidget as _QTableWidget,
    QApplication as _QApplication,
    QDoubleSpinBox as _QDoubleSpinBox,
    QTableWidgetItem as _QTableWidgetItem,
    QAbstractItemView as _QAbstractItemView,
    )
from qtpy.QtGui import (
    QFont as _QFont,
    QIcon as _QIcon,
    QColor as _QColor,
    QBrush as _QBrush,
    QPixmap as _QPixmap,
    QStandardItemModel as _QStandardItemModel,
    )
from qtpy.QtCore import (
    Qt as _Qt,
    QSize as _QSize,
    QTimer as _QTimer,
    Signal as _Signal,
    )
from matplotlib.figure import Figure as _Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as _FigureCanvas,
    NavigationToolbar2QT as _Toolbar
    )

import hallbench.gui.utils as _utils


_font = _QFont()
_font.setPointSize(11)
_font.setBold(False)

_font_bold = _QFont()
_font_bold.setPointSize(11)
_font_bold.setBold(True)


class CheckableComboBox(_QComboBox):
    """Combo box with checkable items."""

    def __init__(self, parent=None):
        """Initialize object."""
        super().__init__(parent)
        self.setFont(_font)
        self.setView(_QListView(self))
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(_QStandardItemModel(self))

    def handleItemPressed(self, index):
        """Change item check state."""
        item = self.model().itemFromIndex(index)
        if item.checkState() == _Qt.Checked:
            item.setCheckState(_Qt.Unchecked)
        else:
            item.setCheckState(_Qt.Checked)

    def checkedItems(self):
        """Get checked items."""
        checkedItems = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                checkedItems.append(item)
        return checkedItems

    def checkedIndexes(self):
        """Get checked indexes."""
        checkedIndexes = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                checkedIndexes.append(index)
        return checkedIndexes

    def checkedItemsText(self):
        """Get checked items text."""
        checkedItemsText = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                checkedItemsText.append(item.text())
        return checkedItemsText


class CurrentPositionWidget(_QWidget):
    """Current Position Widget class for the Hall Bench Control application."""

    _list_of_axis = [1, 2, 3, 5, 6, 7, 8, 9]
    _list_of_axis_names = ["Z", "Y", "X", "A", "W", "V", "B", "C"]
    _list_of_axis_units = ["mm", "mm", "mm", "deg", "mm", "mm", "deg", "deg"]
    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)
        self.setWindowTitle("Position")
        self.resize(222, 282)
        self.setFont(_font)

        main_layout = _QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        grid_layout =_QGridLayout()
        grid_layout.setContentsMargins(6, 6, 6, 6)

        group_box = _QGroupBox("Current Position")
        group_box.setFont(_font_bold)
        group_box.setLayout(grid_layout)

        for idx, axis in enumerate(self._list_of_axis):
            ax_name = self._list_of_axis_names[idx]
            ax_unit = self._list_of_axis_units[idx]

            la_ax_name = _QLabel("#{0:d} (+{1:s}):".format(axis, ax_name))
            la_ax_name.setFont(_font)

            le_ax = _QLineEdit()
            le_ax.setMinimumSize(_QSize(110, 0))
            le_ax.setFont(_font)
            le_ax.setText("")
            le_ax.setAlignment(
                _Qt.AlignRight|_Qt.AlignTrailing|_Qt.AlignVCenter)
            le_ax.setReadOnly(True)
            setattr(self, 'le_posax' + str(axis), le_ax)

            la_ax_unit = _QLabel(ax_unit)
            la_ax_unit.setFont(_font)

            grid_layout.addWidget(la_ax_name, idx, 0, 1, 1)
            grid_layout.addWidget(le_ax, idx, 1, 1, 1)
            grid_layout.addWidget(la_ax_unit, idx, 2, 1, 1)

        main_layout.addWidget(group_box)
        self.setLayout(main_layout)

        self.timer = _QTimer()
        self.timer.timeout.connect(self.updatePositions)
        self.timer.start(self._timer_interval)

    @property
    def positions(self):
        """Return current posiitons dict."""
        return _QApplication.instance().positions

    def closeEvent(self, event):
        """Stop timer and close widget."""
        try:
            self.timer.stop()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def updatePositions(self):
        """Update positions."""
        try:
            if not self.isVisible():
                return

            for axis in self._list_of_axis:
                le = getattr(self, 'le_posax' + str(axis))
                if axis in self.positions:
                    pos = self.positions[axis]
                    le.setText('{0:0.4f}'.format(pos))
                else:
                    le.setText('')
        except Exception:
            pass


class InterpolationTableDialog(_QDialog):
    """Interpolation table class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)
        self.setWindowTitle("Calibration Data Table")
        self.resize(1000, 460)
        self.setFont(_font)

        main_layout = _QHBoxLayout()
        main_layout.setSpacing(20)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('copy')), _QIcon.Normal, _QIcon.Off)

        sensors = ["X", "Y", "Z"]
        for idx, sensor in enumerate(sensors):
            vertical_layout = _QVBoxLayout()
            horizontal_layout = _QHBoxLayout()

            group_box = _QGroupBox("Sensor {0:s}".format(sensor.upper()))
            group_box.setFont(_font_bold)
            group_box.setAlignment(
                _Qt.AlignLeading|_Qt.AlignLeft|_Qt.AlignVCenter)
            group_box.setLayout(vertical_layout)

            table = _QTableWidget()
            table.setFont(_font)
            table.setEditTriggers(_QAbstractItemView.NoEditTriggers)
            table.setColumnCount(2)
            table.setRowCount(0)
            table.setHorizontalHeaderItem(
                0, _QTableWidgetItem("Voltage [V]"))
            table.setHorizontalHeaderItem(
                1, _QTableWidgetItem("Magnetic Field [T]"))
            table.horizontalHeader().setDefaultSectionSize(130)
            table.horizontalHeader().setMinimumSectionSize(130)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setHighlightSections(True)
            setattr(self, 'sensor' + sensor.lower() + '_ta', table)

            label = _QLabel("Display Precision:")
            label.setFont(_font)

            spin_box = _QSpinBox()
            spin_box.setFont(_font)
            spin_box.setValue(4)
            setattr(self, 'sb_sensor' + sensor.lower() + '_prec', spin_box)

            tbt_copy = _QToolButton()
            tbt_copy.setIcon(icon)
            tbt_copy.setIconSize(_QSize(24, 24))
            setattr(self, 'pbt_sensor' + sensor.lower() + '_copy', tbt_copy)

            spacer_item = _QSpacerItem(
                40, 20, _QSizePolicy.Expanding, _QSizePolicy.Minimum)

            vertical_layout.addWidget(table)
            horizontal_layout.addWidget(label)
            horizontal_layout.addWidget(spin_box)
            horizontal_layout.addWidget(tbt_copy)
            horizontal_layout.addItem(spacer_item)
            vertical_layout.addLayout(horizontal_layout)
            main_layout.addWidget(group_box)

        self.setLayout(main_layout)

        self.local_hall_probe = None
        self.clip = _QApplication.clipboard()

        # create connections
        self.pbt_sensorx_copy.clicked.connect(
            lambda: self.copyToClipboard('x'))
        self.pbt_sensory_copy.clicked.connect(
            lambda: self.copyToClipboard('y'))
        self.pbt_sensorz_copy.clicked.connect(
            lambda: self.copyToClipboard('z'))

        self.sb_sensorx_prec.valueChanged.connect(self.updateTablesensorX)
        self.sb_sensory_prec.valueChanged.connect(self.updateTablesensorY)
        self.sb_sensorz_prec.valueChanged.connect(self.updateTablesensorZ)

    def _updateTable(self, table, data, precision):
        table.setRowCount(0)
        formatstr = '{0:0.%if}' % precision
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))

    def clear(self):
        """Clear tables."""
        self.local_hall_probe = None
        self.sensorx_ta.clearContents()
        self.sensorx_ta.setRowCount(0)
        self.sensory_ta.clearContents()
        self.sensory_ta.setRowCount(0)
        self.sensorz_ta.clearContents()
        self.sensorz_ta.setRowCount(0)

    def copyToClipboard(self, sensor):
        """Copy table data to clipboard."""
        table = getattr(self, 'sensor' + sensor + '_ta')
        text = ""
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                text += str(table.item(r, c).text()) + "\t"
            text = text[:-1] + "\n"
        self.clip.setText(text)

    def show(self, hall_probe):
        """Update hall probe object and show dialog."""
        self.local_hall_probe = hall_probe
        self.updateTables()
        super(InterpolationTableDialog, self).show()

    def updateTables(self):
        """Update table values."""
        if self.local_hall_probe is None:
            return
        self.updateTablesensorX()
        self.updateTablesensorY()
        self.updateTablesensorZ()

    def updateTablesensorX(self):
        """Update sensor x table values."""
        precision = self.sb_sensorx_prec.value()
        table = self.sensorx_ta

        if self.local_hall_probe.sensorx is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorx.data
        self._updateTable(table, data, precision)

    def updateTablesensorY(self):
        """Update sensor y table values."""
        precision = self.sb_sensory_prec.value()
        table = self.sensory_ta

        if self.local_hall_probe.sensory is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensory.data
        self._updateTable(table, data, precision)

    def updateTablesensorZ(self):
        """Update sensor z table values."""
        precision = self.sb_sensorz_prec.value()
        table = self.sensorz_ta

        if self.local_hall_probe.sensorz is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorz.data
        self._updateTable(table, data, precision)


class MoveAxisWidget(_QWidget):
    """Move axis widget class for the Hall Bench Control application."""

    _axis_unit = {
        1: 'mm', 2: 'mm', 3: 'mm', 5: 'deg',
        6: 'mm', 7: 'mm', 8: 'deg', 9: 'deg',
    }
    _position_format = '{0:0.3f}'

    def __init__(self, parent=None):
        """Set up the ui, add position widget and create connections."""
        super().__init__(parent)
        self.setWindowTitle("Move Axis")
        self.resize(260, 570)
        self.setFont(_font)

        size_policy = _QSizePolicy(
            _QSizePolicy.Maximum, _QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)

        main_layout = _QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        grid_layout = _QGridLayout()
        grid_layout.setContentsMargins(9, 9, 9, 9)
        grid_layout.setSpacing(6)

        self.current_position_widget = CurrentPositionWidget(self)
        self.current_position_widget.setSizePolicy(size_policy)
        self.current_position_widget.setMinimumSize(_QSize(270, 300))
        main_layout.addWidget(self.current_position_widget)

        self.gb_moveaxis = _QGroupBox("Move Axis")
        self.gb_moveaxis.setFont(_font_bold)
        self.gb_moveaxis.setLayout(grid_layout)

        label = _QLabel("Axis:")
        label.setFont(_font)
        grid_layout.addWidget(label, 0, 0, 1, 1)

        self.cmb_selectaxis = _QComboBox()
        self.cmb_selectaxis.setFont(_font)
        self.cmb_selectaxis.addItem("")
        self.cmb_selectaxis.addItem("#1 (+Z)")
        self.cmb_selectaxis.addItem("#2 (+Y)")
        self.cmb_selectaxis.addItem("#3 (+X)")
        self.cmb_selectaxis.addItem("#5 (+A)")
        self.cmb_selectaxis.addItem("#6 (+W)")
        self.cmb_selectaxis.addItem("#7 (+V)")
        self.cmb_selectaxis.addItem("#8 (+B)")
        self.cmb_selectaxis.addItem("#9 (+C)")
        grid_layout.addWidget(self.cmb_selectaxis, 0, 1, 1, 1)

        label = _QLabel("Velocity:")
        label.setFont(_font)
        grid_layout.addWidget(label, 1, 0, 1, 1)

        self.le_targetvel = _QLineEdit()
        self.le_targetvel.setSizePolicy(size_policy)
        self.le_targetvel.setFont(_font)
        grid_layout.addWidget(self.le_targetvel, 1, 1, 1, 1)

        self.la_targetvelunit = _QLabel("")
        self.la_targetvelunit.setFont(_font)
        grid_layout.addWidget(self.la_targetvelunit, 1, 2, 1, 1)

        label = _QLabel("Position:")
        label.setFont(_font)
        grid_layout.addWidget(label, 2, 0, 1, 1)

        self.le_targetpos = _QLineEdit()
        self.le_targetpos.setSizePolicy(size_policy)
        self.le_targetpos.setFont(_font)
        grid_layout.addWidget(self.le_targetpos, 2, 1, 1, 1)

        self.la_targetposunit = _QLabel("")
        self.la_targetposunit.setFont(_font)
        grid_layout.addWidget(self.la_targetposunit, 2, 2, 1, 1)

        vertical_layout = _QVBoxLayout()
        vertical_layout.setSpacing(15)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('move')), _QIcon.Normal, _QIcon.Off)
        self.pbt_move = _QPushButton("Move to Target")
        self.pbt_move.setIcon(icon)
        self.pbt_move.setMinimumSize(_QSize(200, 60))
        vertical_layout.addWidget(self.pbt_move)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('stop')), _QIcon.Normal, _QIcon.Off)
        self.pbt_stop = _QPushButton("Stop Motor")
        self.pbt_stop.setIcon(icon)
        self.pbt_stop.setMinimumSize(_QSize(200, 60))
        self.pbt_stop.setObjectName("pbt_stop")
        vertical_layout.addWidget(self.pbt_stop)

        grid_layout.addLayout(vertical_layout, 3, 0, 1, 3)
        main_layout.addWidget(self.gb_moveaxis)
        self.setLayout(main_layout)

        self.connectSignalSlots()

    @property
    def pmac(self):
        """Pmac communication class."""
        return _QApplication.instance().devices.pmac

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.le_targetvel.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.le_targetvel))
        self.le_targetpos.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.le_targetpos))

        self.cmb_selectaxis.currentIndexChanged.connect(
            self.updateVelocityAndPosition)

        self.pbt_move.clicked.connect(self.moveToTarget)
        self.pbt_stop.clicked.connect(self.stopAxis)

    def moveToTarget(self, axis):
        """Move axis to target position."""
        try:
            targetpos_str = self.le_targetpos.text()
            targetvel_str = self.le_targetvel.text()

            if len(targetpos_str) == 0 or len(targetvel_str) == 0:
                return

            targetpos = _utils.getValueFromStringExpresssion(
                self.le_targetpos.text())
            targetvel = _utils.getValueFromStringExpresssion(
                self.le_targetvel.text())

            axis = self.selectedAxis()
            if axis is None:
                return

            velocity = self.pmac.get_velocity(axis)

            if targetvel != velocity:
                self.pmac.set_axis_speed(axis, targetvel)

            self.pmac.move_axis(axis, targetpos)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to move axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def selectedAxis(self):
        """Return the selected axis."""
        axis_str = self.cmb_selectaxis.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in self.pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def setVelocityPositionStrFormat(self, line_edit):
        """Set the velocity and position string format."""
        try:
            if not _utils.setFloatLineEditText(line_edit, precision=3):
                self.updateVelocityAndPosition()
        except Exception:
            pass

    def stopAxis(self):
        """Stop the selected axis."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return
            self.pmac.stop_axis(axis)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def updateVelocityAndPosition(self):
        """Update velocity and position values for the selected axis."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                self.le_targetvel.setText('')
                self.le_targetpos.setText('')
                self.la_targetvelunit.setText('')
                self.la_targetposunit.setText('')
                return

            velocity = self.pmac.get_velocity(axis)
            position = self.pmac.get_position(axis)
            self.le_targetvel.setText(self._position_format.format(
                velocity))
            self.le_targetpos.setText(self._position_format.format(
                position))

            self.la_targetvelunit.setText(self._axis_unit[axis] + '/s')
            self.la_targetposunit.setText(self._axis_unit[axis])
        except Exception:
            pass


class PlotDialog(_QDialog):
    """Matplotlib plot dialog."""

    def __init__(self, parent=None):
        """Add figure canvas to layout."""
        super().__init__(parent)
        self.setFont(_font)
        self.figure = _Figure()
        self.canvas = _FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.canvas)
        self.toolbar = _Toolbar(self.canvas, self)
        _layout.addWidget(self.toolbar)
        self.setLayout(_layout)

    def updatePlot(self):
        """Update plot."""
        self.canvas.draw()

    def show(self):
        """Show dialog."""
        self.updatePlot()
        super().show()


class PolynomialTableDialog(_QDialog):
    """Polynomial table class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)
        self.setWindowTitle("Calibration Data Table")
        self.resize(1000, 650)
        self.setFont(_font)

        main_layout = _QVBoxLayout()
        main_layout.setSpacing(20)

        label = _QLabel(
            ("Magnetic Field [T] = C0 + C1*Voltage[V]" +
             " + C2*Voltage[V]² + C3*Voltage[V]³ + ... "))
        label.setAlignment(_Qt.AlignCenter)
        main_layout.addWidget(label)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('copy')), _QIcon.Normal, _QIcon.Off)

        sensors = ["X", "Y", "Z"]
        for idx, sensor in enumerate(sensors):
            vertical_layout = _QVBoxLayout()
            form_layout = _QFormLayout()
            horizontal_layout = _QHBoxLayout()

            group_box = _QGroupBox("Sensor {0:s}".format(sensor.upper()))
            group_box.setFont(_font_bold)
            group_box.setLayout(vertical_layout)

            label_volt = _QLabel("Voltage Interval")
            size_policy = _QSizePolicy(
                _QSizePolicy.Fixed, _QSizePolicy.Preferred)
            size_policy.setHorizontalStretch(0)
            size_policy.setVerticalStretch(0)
            label_volt.setSizePolicy(size_policy)
            label_volt.setMinimumSize(_QSize(260, 0))
            label_volt.setMaximumSize(_QSize(260, 16777215))
            label_volt.setAlignment(_Qt.AlignCenter)

            label_poly = _QLabel("Polynomial Coefficients")
            label_poly.setAlignment(_Qt.AlignCenter)

            table = _QTableWidget()
            table.setFont(_font)
            table.setEditTriggers(_QAbstractItemView.NoEditTriggers)
            table.setColumnCount(6)
            table.setRowCount(0)
            table.setHorizontalHeaderItem(
                0, _QTableWidgetItem("Initial Voltage [V]"))
            table.setHorizontalHeaderItem(
                1, _QTableWidgetItem("Final Voltage [V]"))
            table.setHorizontalHeaderItem(
                2, _QTableWidgetItem("C0"))
            table.setHorizontalHeaderItem(
                3, _QTableWidgetItem("C1"))
            table.setHorizontalHeaderItem(
                4, _QTableWidgetItem("C2"))
            table.setHorizontalHeaderItem(
                5, _QTableWidgetItem("C3"))
            table.horizontalHeader().setDefaultSectionSize(120)
            table.horizontalHeader().setMinimumSectionSize(120)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(False)
            setattr(self, 'sensor' + sensor.lower() + '_ta', table)

            label = _QLabel("Display Precision:")
            label.setFont(_font)

            spin_box = _QSpinBox()
            spin_box.setFont(_font)
            spin_box.setValue(4)
            setattr(self, 'sb_sensor' + sensor.lower() + '_prec', spin_box)

            tbt_copy = _QToolButton()
            tbt_copy.setIcon(icon)
            tbt_copy.setIconSize(_QSize(24, 24))
            setattr(self, 'pbt_sensor' + sensor.lower() + '_copy', tbt_copy)

            spacer_item = _QSpacerItem(
                40, 20, _QSizePolicy.Expanding, _QSizePolicy.Minimum)

            form_layout.setWidget(0, _QFormLayout.LabelRole, label_volt)
            form_layout.setWidget(0, _QFormLayout.FieldRole, label_poly)

            horizontal_layout.addWidget(label)
            horizontal_layout.addWidget(spin_box)
            horizontal_layout.addWidget(tbt_copy)
            horizontal_layout.addItem(spacer_item)

            vertical_layout.addLayout(form_layout)
            vertical_layout.addWidget(table)
            vertical_layout.addLayout(horizontal_layout)

            main_layout.addWidget(group_box)

        self.setLayout(main_layout)

        self.clip = _QApplication.clipboard()
        self.local_hall_probe = None

        # create connections
        self.pbt_sensorx_copy.clicked.connect(
            lambda: self.copyToClipboard('x'))
        self.pbt_sensory_copy.clicked.connect(
            lambda: self.copyToClipboard('y'))
        self.pbt_sensorz_copy.clicked.connect(
            lambda: self.copyToClipboard('z'))

        self.sb_sensorx_prec.valueChanged.connect(self.updateTableSensorX)
        self.sb_sensory_prec.valueChanged.connect(self.updateTableSensorY)
        self.sb_sensorz_prec.valueChanged.connect(self.updateTableSensorZ)

    def _updateTable(self, table, data, precision):
        table.setRowCount(0)

        if len(data) == 0:
            return

        nc = len(data[0])
        table.setColumnCount(nc)
        labels = ['Initial Voltage [V]', 'Final Voltage [V]']
        for j in range(nc-2):
            labels.append('C' + str(j))
        table.setHorizontalHeaderLabels(labels)

        vformatstr = '{0:0.%if}' % precision
        cformatstr = '{0:0.%ie}' % precision
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                if j < 2:
                    table.setItem(i, j, _QTableWidgetItem(
                        vformatstr.format(row[j])))
                else:
                    table.setItem(i, j, _QTableWidgetItem(
                        cformatstr.format(row[j])))

    def clear(self):
        """Clear tables."""
        self.local_hall_probe = None
        self.sensorx_ta.clearContents()
        self.sensorx_ta.setRowCount(0)
        self.sensory_ta.clearContents()
        self.sensory_ta.setRowCount(0)
        self.sensorz_ta.clearContents()
        self.sensorz_ta.setRowCount(0)

    def copyToClipboard(self, sensor):
        """Copy table data to clipboard."""
        table = getattr(self, 'sensor' + sensor + '_ta')
        text = ""
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                text += str(table.item(r, c).text()) + "\t"
            text = text[:-1] + "\n"
        self.clip.setText(text)

    def show(self, hall_probe=None):
        """Update hall probe object and show dialog."""
        self.local_hall_probe = hall_probe
        self.updateTables()
        super().show()

    def updateTables(self):
        """Update table values."""
        if self.local_hall_probe is None:
            return
        self.updateTableSensorX()
        self.updateTableSensorY()
        self.updateTableSensorZ()

    def updateTableSensorX(self):
        """Update sensor x table values."""
        precision = self.sb_sensorx_prec.value()
        table = self.sensorx_ta
        if self.local_hall_probe.sensorx is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorx.data
        self._updateTable(table, data, precision)

    def updateTableSensorY(self):
        """Update sensor y table values."""
        precision = self.sb_sensory_prec.value()
        table = self.sensory_ta
        if self.local_hall_probe.sensory is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensory.data
        self._updateTable(table, data, precision)

    def updateTableSensorZ(self):
        """Update sensor z table values."""
        precision = self.sb_sensorz_prec.value()
        table = self.sensorz_ta
        if self.local_hall_probe.sensorz is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorz.data
        self._updateTable(table, data, precision)


class PreferencesDialog(_QDialog):
    """Preferences dialog class for Hall Bench Control application."""

    preferences_changed = _Signal([dict])

    def __init__(self, chb_names, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(250, 400)
        self.setFont(_font)

        main_layout = _QVBoxLayout()
        vertical_layout = _QVBoxLayout()
        group_box = _QGroupBox("Select Tabs to Show")
        group_box.setLayout(vertical_layout)
        group_box.setFont(_font_bold)
        main_layout.addWidget(group_box)
        self.setLayout(main_layout)

        self.chb_names = chb_names
        for name in self.chb_names:
            name_split = name.split('_')
            label = ' '.join([s.capitalize() for s in name_split])
            chb = _QCheckBox(label)
            setattr(self, 'chb_' + name, chb)
            vertical_layout.addWidget(chb)
            chb.setFont(_font)

        self.pbt_apply = _QPushButton("Apply Changes")
        self.pbt_apply.setMinimumSize(_QSize(0, 40))
        self.pbt_apply.setFont(_font_bold)
        vertical_layout.addWidget(self.pbt_apply)

        self.pbt_apply.clicked.connect(self.tabsPreferencesChanged)
        self.chb_connection.setChecked(True)
        self.chb_motors.setChecked(True)
        self.chb_measurement.setChecked(True)

    def tabsPreferencesChanged(self):
        """Get tabs checkbox status and emit signal to change tabs."""
        try:
            chb_status = {}
            for chb_name in self.chb_names:
                chb = getattr(self, 'chb_' + chb_name)
                chb_status[chb_name] = chb.isChecked()

            self.preferences_changed.emit(chb_status)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class TableAnalysisDialog(_QDialog):
    """Table data analysis dialog class."""

    def __init__(self, parent=None):
        """Add table widget and copy button."""
        super().__init__(parent)
        self.setFont(_font)

        self.setWindowTitle("Statistics")
        self.results_ta = _QTableWidget()
        self.results_ta.setAlternatingRowColors(True)
        self.results_ta.horizontalHeader().setStretchLastSection(True)
        self.results_ta.horizontalHeader().setDefaultSectionSize(120)

        self.pbt_copy = _QPushButton("Copy to clipboard")
        self.pbt_copy.clicked.connect(self.copyToClipboard)
        self.pbt_copy.setFont(_font_bold)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.results_ta)
        _layout.addWidget(self.pbt_copy)
        self.setLayout(_layout)
        self.table_df = None

        self.resize(500, 200)

    def addItemsToTable(self, text, i, j):
        """Add items to table."""
        item = _QTableWidgetItem(text)
        item.setFlags(_Qt.ItemIsSelectable | _Qt.ItemIsEnabled)
        self.results_ta.setItem(i, j, item)

    def analyseAndShowResults(self):
        """Analyse data and add results to table."""
        self.results_ta.clearContents()
        self.results_ta.setRowCount(0)
        self.results_ta.setColumnCount(0)

        if self.table_df is None:
            return

        self.results_ta.setColumnCount(3)

        self.results_ta.setHorizontalHeaderLabels(
            ['Mean', 'STD', 'Peak-Valey'])

        labels = [
            l for l in self.table_df.columns if l not in ['Date', 'Time']]

        self.results_ta.setRowCount(len(labels))
        self.results_ta.setVerticalHeaderLabels(labels)

        for i in range(len(labels)):
            label = labels[i]
            values = self.table_df[label].values
            try:
                values = values.astype(float)
            except Exception:
                values = [_np.nan]*len(values)
            values = _np.array(values)
            values = values[_np.isfinite(values)]
            if len(values) == 0:
                mean = _np.nan
                std = _np.nan
                peak_valey = _np.nan
            else:
                mean = _np.mean(values)
                std = _np.std(values)
                peak_valey = _np.max(values) - _np.min(values)
            self.addItemsToTable('{0:.4f}'.format(mean), i, 0)
            self.addItemsToTable('{0:.4f}'.format(std), i, 1)
            self.addItemsToTable('{0:.4f}'.format(peak_valey), i, 2)

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def clear(self):
        """Clear data and table."""
        self.table_df = None
        self.results_ta.clearContents()
        self.results_ta.setRowCount(0)
        self.results_ta.setColumnCount(0)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = _utils.tableToDataFrame(self.results_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def show(self, table_df):
        """Show dialog."""
        self.table_df = table_df
        self.analyseAndShowResults()
        super().show()

    def updateData(self, table_df):
        """Update table data."""
        self.table_df = table_df
        self.analyseAndShowResults()


class TableDialog(_QDialog):
    """Table dialog class."""

    def __init__(self, parent=None):
        """Add table widget and copy button."""
        super().__init__(parent)
        self.resize(800, 500)
        self.setFont(_font)

        self.setWindowTitle("Data Table")
        self.data_ta = _QTableWidget()
        self.data_ta.setAlternatingRowColors(True)
        self.data_ta.verticalHeader().hide()
        self.data_ta.horizontalHeader().setStretchLastSection(True)
        self.data_ta.horizontalHeader().setDefaultSectionSize(120)

        self.pbt_copy = _QPushButton("Copy to clipboard")
        self.pbt_copy.clicked.connect(self.copyToClipboard)
        self.pbt_copy.setFont(_font_bold)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.data_ta)
        _layout.addWidget(self.pbt_copy)
        self.setLayout(_layout)
        self.table_df = None

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def addItemsToTable(self, text, i, j):
        """Add items to table."""
        item = _QTableWidgetItem(text)
        item.setFlags(_Qt.ItemIsSelectable | _Qt.ItemIsEnabled)
        self.data_ta.setItem(i, j, item)

    def clear(self):
        """Clear data and table."""
        self.table_df = None
        self.data_ta.clearContents()
        self.data_ta.setRowCount(0)
        self.data_ta.setColumnCount(0)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = _utils.tableToDataFrame(self.data_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def show(self, table_df):
        """Show dialog."""
        self.table_df = table_df
        self.updateTable()
        super().show()

    def updateData(self, table_df):
        """Update table data."""
        self.table_df = table_df
        self.updateTable()

    def updateTable(self):
        """Add data to table."""
        self.data_ta.clearContents()
        self.data_ta.setRowCount(0)
        self.data_ta.setColumnCount(0)

        if self.table_df is None:
            return

        nrows = self.table_df.shape[0]
        ncols = self.table_df.shape[1]

        self.data_ta.setRowCount(nrows)
        self.data_ta.setColumnCount(ncols)

        columns = self.table_df.columns.values
        self.data_ta.setHorizontalHeaderLabels(columns)

        for i in range(nrows):
            for j in range(ncols):
                if columns[j] == 'ID':
                    text = '{0:d}'.format(int(self.table_df.iloc[i, j]))
                else:
                    text = str(self.table_df.iloc[i, j])
                self.addItemsToTable(text, i, j)


class TablePlotWidget(_QWidget):
    """Table and Plot widget."""

    _left_axis_1_label = ''
    _right_axis_1_label = ''
    _right_axis_2_label = ''

    _left_axis_1_format = '{0:.4f}'
    _right_axis_1_format = '{0:.4f}'
    _right_axis_2_format = '{0:.4f}'

    _left_axis_1_data_labels = []
    _right_axis_1_data_labels = []
    _right_axis_2_data_labels = []

    _left_axis_1_data_colors = []
    _right_axis_1_data_colors = []
    _right_axis_2_data_colors = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Table and Plot")
        self.resize(1230, 900)
        self.addWidgets()
        self.setFont(_font)

        # variables initialisation
        self._timestamp = []
        self._legend_items = []
        self._graphs = {}
        self._data_labels = (
            self._left_axis_1_data_labels +
            self._right_axis_1_data_labels +
            self._right_axis_2_data_labels)
        self._data_formats = (
            [self._left_axis_1_format]*len(self._left_axis_1_data_labels) +
            [self._right_axis_1_format]*len(self._right_axis_1_data_labels) +
            [self._right_axis_2_format]*len(self._right_axis_2_data_labels))
        self._readings = {}
        for i, label in enumerate(self._data_labels):
            self._readings[label] = []

        # create timer to monitor values
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(lambda: self.readValue(monitor=True))

        # create table analysis dialog
        self.table_analysis_dialog = TableAnalysisDialog()

        # add legend to plot
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.pw_plot.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.right_axis_1 = None
        self.right_axis_2 = None
        self.configurePlot()
        self.configureTable()
        self.connectSignalSlots()

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    def addLastValueToTable(self):
        """Add the last value read to table."""
        if len(self._timestamp) == 0:
            return

        n = self.table_ta.rowCount() + 1
        self.table_ta.setRowCount(n)

        dt = _datetime.datetime.fromtimestamp(self._timestamp[-1])
        date = dt.strftime("%d/%m/%Y")
        hour = dt.strftime("%H:%M:%S")
        self.table_ta.setItem(n-1, 0, _QTableWidgetItem(date))
        self.table_ta.setItem(n-1, 1, _QTableWidgetItem(hour))

        for j, label in enumerate(self._data_labels):
            fmt = self._data_formats[j]
            reading = self._readings[label][-1]
            self.table_ta.setItem(
                n-1, j+2, _QTableWidgetItem(fmt.format(reading)))

        vbar = self.table_ta.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def addWidgets(self):
        """Add widgets and layouts."""
        icon_size = _QSize(24, 24)

        # Layouts
        self.vertical_layout_1 = _QVBoxLayout()
        self.vertical_layout_2 = _QVBoxLayout()
        self.vertical_layout_2.setSpacing(20)
        self.vertical_layout_3 = _QVBoxLayout()
        self.horizontal_layout_1 = _QHBoxLayout()
        self.horizontal_layout_2 = _QHBoxLayout()
        self.horizontal_layout_3 = _QHBoxLayout()
        self.horizontal_layout_4 = _QHBoxLayout()

        # Plot Widget
        self.pw_plot = _pyqtgraph.PlotWidget()
        brush = _QBrush(_QColor(255, 255, 255))
        brush.setStyle(_Qt.NoBrush)
        self.pw_plot.setBackgroundBrush(brush)
        self.pw_plot.setForegroundBrush(brush)
        self.horizontal_layout_1.addWidget(self.pw_plot)

        # Read button
        self.pbt_read = _QPushButton("Read")
        self.pbt_read.setMinimumSize(_QSize(0, 45))
        self.pbt_read.setFont(_font_bold)
        self.vertical_layout_2.addWidget(self.pbt_read)

        # Monitor button
        self.pbt_monitor = _QPushButton("Monitor")
        self.pbt_monitor.setMinimumSize(_QSize(0, 45))
        self.pbt_monitor.setFont(_font_bold)
        self.pbt_monitor.setCheckable(True)
        self.pbt_monitor.setChecked(False)
        self.vertical_layout_2.addWidget(self.pbt_monitor)

        # Monitor step
        label = _QLabel("Step")
        label.setAlignment(_Qt.AlignRight|_Qt.AlignTrailing|_Qt.AlignVCenter)
        self.horizontal_layout_3.addWidget(label)

        self.sbd_monitorstep = _QDoubleSpinBox()
        self.sbd_monitorstep.setDecimals(1)
        self.sbd_monitorstep.setMinimum(0.1)
        self.sbd_monitorstep.setMaximum(60.0)
        self.sbd_monitorstep.setProperty("value", 10.0)
        self.horizontal_layout_3.addWidget(self.sbd_monitorstep)

        self.cmb_monitorunit = _QComboBox()
        self.cmb_monitorunit.addItem("sec")
        self.cmb_monitorunit.addItem("min")
        self.cmb_monitorunit.addItem("hour")
        self.horizontal_layout_3.addWidget(self.cmb_monitorunit)
        self.vertical_layout_2.addLayout(self.horizontal_layout_3)

        # Group box with read and monitor buttons
        self.group_box = _QGroupBox()
        self.group_box.setMinimumSize(_QSize(270, 0))
        self.group_box.setTitle("")
        self.group_box.setLayout(self.vertical_layout_2)

        # Table widget
        self.table_ta = _QTableWidget()
        sizePolicy = _QSizePolicy(
            _QSizePolicy.Expanding, _QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.table_ta.sizePolicy().hasHeightForWidth())
        self.table_ta.setSizePolicy(sizePolicy)
        self.table_ta.setVerticalScrollBarPolicy(_Qt.ScrollBarAlwaysOn)
        self.table_ta.setHorizontalScrollBarPolicy(_Qt.ScrollBarAsNeeded)
        self.table_ta.setEditTriggers(_QAbstractItemView.NoEditTriggers)
        self.table_ta.setColumnCount(0)
        self.table_ta.setRowCount(0)
        self.table_ta.horizontalHeader().setVisible(True)
        self.table_ta.horizontalHeader().setCascadingSectionResizes(False)
        self.table_ta.horizontalHeader().setDefaultSectionSize(200)
        self.table_ta.horizontalHeader().setHighlightSections(True)
        self.table_ta.horizontalHeader().setMinimumSectionSize(80)
        self.table_ta.horizontalHeader().setStretchLastSection(True)
        self.table_ta.verticalHeader().setVisible(False)
        self.horizontal_layout_4.addWidget(self.table_ta)

        # Tool buttons
        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('font')), _QIcon.Normal, _QIcon.Off)
        self.tbt_autorange = _QToolButton()
        self.tbt_autorange.setIcon(icon)
        self.tbt_autorange.setIconSize(icon_size)
        self.tbt_autorange.setCheckable(True)
        self.tbt_autorange.setToolTip('Turn on plot autorange.')
        self.vertical_layout_3.addWidget(self.tbt_autorange)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('save')), _QIcon.Normal, _QIcon.Off)
        self.tbt_save = _QToolButton()
        self.tbt_save.setIcon(icon)
        self.tbt_save.setIconSize(icon_size)
        self.tbt_save.setToolTip('Save table data to file.')
        self.vertical_layout_3.addWidget(self.tbt_save)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('copy')), _QIcon.Normal, _QIcon.Off)
        self.tbt_copy = _QToolButton()
        self.tbt_copy.setIcon(icon)
        self.tbt_copy.setIconSize(icon_size)
        self.tbt_copy.setToolTip('Copy table data.')
        self.vertical_layout_3.addWidget(self.tbt_copy)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('stats')), _QIcon.Normal, _QIcon.Off)
        self.pbt_stats = _QToolButton()
        self.pbt_stats.setIcon(icon)
        self.pbt_stats.setIconSize(icon_size)
        self.pbt_stats.setToolTip('Show data statistics.')
        self.vertical_layout_3.addWidget(self.pbt_stats)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('delete')), _QIcon.Normal, _QIcon.Off)
        self.pbt_remove = _QToolButton()
        self.pbt_remove.setIcon(icon)
        self.pbt_remove.setIconSize(icon_size)
        self.pbt_remove.setToolTip('Remove selected lines from table.')
        self.vertical_layout_3.addWidget(self.pbt_remove)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.getIconPath('clear')), _QIcon.Normal, _QIcon.Off)
        self.tbt_clear = _QToolButton()
        self.tbt_clear.setIcon(icon)
        self.tbt_clear.setIconSize(icon_size)
        self.tbt_clear.setToolTip('Clear table data.')
        self.vertical_layout_3.addWidget(self.tbt_clear)

        spacer_item = _QSpacerItem(
            20, 100, _QSizePolicy.Minimum, _QSizePolicy.Fixed)
        self.vertical_layout_3.addItem(spacer_item)
        self.horizontal_layout_4.addLayout(self.vertical_layout_3)

        self.horizontal_layout_2.addWidget(self.group_box)
        self.horizontal_layout_2.addLayout(self.horizontal_layout_4)
        self.vertical_layout_1.addLayout(self.horizontal_layout_1)
        self.vertical_layout_1.addLayout(self.horizontal_layout_2)
        self.setLayout(self.vertical_layout_1)

    def addWidgetsNextToPlot(self, widget_list):
        """Add widgets on the side of plot widget."""
        if not isinstance(widget_list, (list, tuple)):
            widget_list = [[widget_list]]

        if not isinstance(widget_list[0], (list, tuple)):
            widget_list = [widget_list]

        for idx, lt in enumerate(widget_list):
            _layout = _QVBoxLayout()
            _layout.setContentsMargins(0, 0, 0, 0)
            for wg in lt:
                if isinstance(wg, _QPushButton):
                    wg.setMinimumHeight(45)
                    wg.setFont(_font_bold)
                _layout.addWidget(wg)
            self.horizontal_layout_1.insertLayout(idx, _layout)

    def addWidgetsNextToTable(self, widget_list):
        """Add widgets on the side of table widget."""
        if not isinstance(widget_list, (list, tuple)):
            widget_list = [[widget_list]]

        if not isinstance(widget_list[0], (list, tuple)):
            widget_list = [widget_list]

        for idx, lt in enumerate(widget_list):
            _layout = _QHBoxLayout()
            for wg in lt:
                if isinstance(wg, _QPushButton):
                    wg.setMinimumHeight(45)
                    wg.setFont(_font_bold)
                _layout.addWidget(wg)
            self.vertical_layout_2.insertLayout(idx, _layout)

    def clearLegendItems(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def clearButtonClicked(self):
        """Clear all values."""
        if len(self._timestamp) == 0:
            return

        msg = 'Clear table data?'
        reply = _QMessageBox.question(
            self, 'Message', msg, buttons=_QMessageBox.No|_QMessageBox.Yes,
            defaultButton=_QMessageBox.No)

        if reply == _QMessageBox.Yes:
            self.clear()

    def clear(self):
        """Clear all values."""
        self._timestamp = []
        for label in self._data_labels:
            self._readings[label] = []
        self.updateTableValues()
        self.updatePlot()
        self.updateTableAnalysisDialog()

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.table_analysis_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def configurePlot(self):
        """Configure data plots."""
        self.pw_plot.clear()
        self.pw_plot.setLabel('bottom', 'Time interval [s]')
        self.pw_plot.showGrid(x=True, y=True)

        # Configure left axis 1
        self.pw_plot.setLabel('left', self._left_axis_1_label)

        colors = self._left_axis_1_data_colors
        data_labels = self._left_axis_1_data_labels
        if len(colors) != len(data_labels):
            colors = [(0, 0, 255)]*len(data_labels)

        for i, label in enumerate(data_labels):
            pen = colors[i]
            graph = self.pw_plot.plotItem.plot(
                _np.array([]), _np.array([]), pen=pen, symbol='o',
                symbolPen=pen, symbolSize=3, symbolBrush=pen)
            self._graphs[label] = graph

        # Configure right axis 1
        data_labels = self._right_axis_1_data_labels
        colors = self._right_axis_1_data_colors
        if len(colors) != len(data_labels):
            colors = [(0, 0, 255)]*len(data_labels)

        if len(data_labels) != 0:
            self.right_axis_1 = _utils.plotItemAddFirstRightAxis(
                self.pw_plot.plotItem)
            self.right_axis_1.setLabel(self._right_axis_1_label)
            self.right_axis_1.setStyle(showValues=True)

            for i, label in enumerate(data_labels):
                pen = colors[i]
                graph = _pyqtgraph.PlotDataItem(
                    _np.array([]), _np.array([]), pen=pen, symbol='o',
                    symbolPen=pen, symbolSize=3, symbolBrush=pen)
                self.right_axis_1.linkedView().addItem(graph)
                self._graphs[label] = graph

        # Configure right axis 2
        data_labels = self._right_axis_2_data_labels
        colors = self._right_axis_2_data_colors
        if len(colors) != len(data_labels):
            colors = [(0, 0, 255)]*len(data_labels)

        if len(data_labels) != 0:
            self.right_axis_2 = _utils.plotItemAddSecondRightAxis(
                self.pw_plot.plotItem)
            self.right_axis_2.setLabel(self._right_axis_2_label)
            self.right_axis_2.setStyle(showValues=True)

            for i, label in enumerate(data_labels):
                pen = colors[i]
                graph = _pyqtgraph.PlotDataItem(
                    _np.array([]), _np.array([]), pen=pen, symbol='o',
                    symbolPen=pen, symbolSize=3, symbolBrush=pen)
                self.right_axis_2.linkedView().addItem(graph)
                self._graphs[label] = graph

        # Update legend
        self.updateLegendItems()

    def configureTable(self):
        """Configure table."""
        col_labels = ['Date', 'Time']
        for label in self._data_labels:
            col_labels.append(label)
        self.table_ta.setColumnCount(len(col_labels))
        self.table_ta.setHorizontalHeaderLabels(col_labels)
        self.table_ta.setAlternatingRowColors(True)

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.pbt_read.clicked.connect(lambda: self.readValue(monitor=False))
        self.pbt_monitor.toggled.connect(self.monitorValue)
        self.sbd_monitorstep.valueChanged.connect(self.updateMonitorInterval)
        self.cmb_monitorunit.currentIndexChanged.connect(
            self.updateMonitorInterval)
        self.tbt_autorange.toggled.connect(self.enableAutorange)
        self.tbt_save.clicked.connect(self.saveToFile)
        self.tbt_copy.clicked.connect(self.copyToClipboard)
        self.pbt_stats.clicked.connect(self.showTableAnalysisDialog)
        self.pbt_remove.clicked.connect(self.removeValue)
        self.tbt_clear.clicked.connect(self.clearButtonClicked)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = _utils.tableToDataFrame(self.table_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def enableAutorange(self, checked):
        """Enable or disable plot autorange."""
        if checked:
            if self.right_axis_2 is not None:
                self.right_axis_2.linkedView().enableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            if self.right_axis_1 is not None:
                self.right_axis_1.linkedView().enableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            self.pw_plot.plotItem.enableAutoRange(
                axis=_pyqtgraph.ViewBox.YAxis)
        else:
            if self.right_axis_2 is not None:
                self.right_axis_2.linkedView().disableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            if self.right_axis_1 is not None:
                self.right_axis_1.linkedView().disableAutoRange(
                    axis=_pyqtgraph.ViewBox.YAxis)
            self.pw_plot.plotItem.disableAutoRange(
                axis=_pyqtgraph.ViewBox.YAxis)

    def hideRightAxes(self):
        """Hide right axes."""
        if self.right_axis_1 is not None:
            self.right_axis_1.setStyle(showValues=False)
            self.right_axis_1.setLabel('')
        if self.right_axis_2 is not None:
            self.right_axis_2.setStyle(showValues=False)
            self.right_axis_2.setLabel('')

    def monitorValue(self, checked):
        """Monitor values."""
        if checked:
            self.pbt_read.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.pbt_read.setEnabled(True)

    def readValue(self, monitor=False):
        """Read value."""
        pass

    def removeValue(self):
        """Remove value from list."""
        selected = self.table_ta.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self._timestamp)

        self._timestamp = [
            self._timestamp[i] for i in range(n) if i not in rows]

        for label in self._data_labels:
            readings = self._readings[label]
            self._readings[label] = [
                readings[i] for i in range(n) if i not in rows]

        self.updateTableValues()
        self.updatePlot()
        self.updateTableAnalysisDialog()

    def saveToFile(self):
        """Save table values to file."""
        df = _utils.tableToDataFrame(self.table_ta)
        if df is None:
            _QMessageBox.critical(
                self, 'Failure', 'Empty table.', _QMessageBox.Ok)
            return

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save measurements file.', directory=self.directory,
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if (not filename.endswith('.txt')
               and not filename.endswith('.dat')):
                filename = filename + '.txt'
            df.to_csv(filename, sep='\t')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save data to file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def setTableColumnSize(self, size):
        """Set table horizontal header default section size."""
        self.table_ta.horizontalHeader().setDefaultSectionSize(size)

    def showTableAnalysisDialog(self):
        """Show table analysis dialog."""
        df = _utils.tableToDataFrame(self.table_ta)
        self.table_analysis_dialog.accept()
        self.table_analysis_dialog.show(df)

    def updateLegendItems(self):
        """Update legend items."""
        self.clearLegendItems()
        self._legend_items = []
        for label in self._data_labels:
            legend_label = label.split('[')[0]
            self._legend_items.append(legend_label)
            self.legend.addItem(self._graphs[label], legend_label)

    def updateMonitorInterval(self):
        """Update monitor interval value."""
        index = self.cmb_monitorunit.currentIndex()
        if index == 0:
            mf = 1000
        elif index == 1:
            mf = 1000*60
        else:
            mf = 1000*3600
        self.timer.setInterval(self.sbd_monitorstep.value()*mf)

    def updatePlot(self):
        """Update plot values."""
        if len(self._timestamp) == 0:
            for label in self._data_labels:
                self._graphs[label].setData(
                    _np.array([]), _np.array([]))
            return

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            timeinterval = _np.array(self._timestamp) - self._timestamp[0]
            for label in self._data_labels:
                readings = _np.array(self._readings[label])
                dt = timeinterval[_np.isfinite(readings)]
                rd = readings[_np.isfinite(readings)]
                self._graphs[label].setData(dt, rd)

        if len(self._timestamp) > 2 and self.tbt_autorange.isChecked():
            xmin = timeinterval[0]
            xmax = timeinterval[-1]
            self.pw_plot.plotItem.getViewBox().setRange(xRange=(xmin, xmax))

    def updateTableAnalysisDialog(self):
        """Update table analysis dialog."""
        self.table_analysis_dialog.updateData(
            _utils.tableToDataFrame(self.table_ta))

    def updateTableValues(self):
        """Update table values."""
        n = len(self._timestamp)
        self.table_ta.clearContents()
        self.table_ta.setRowCount(n)

        for i in range(n):
            dt = _datetime.datetime.fromtimestamp(self._timestamp[i])
            date = dt.strftime("%d/%m/%Y")
            hour = dt.strftime("%H:%M:%S")
            self.table_ta.setItem(i, 0, _QTableWidgetItem(date))
            self.table_ta.setItem(i, 1, _QTableWidgetItem(hour))

            for j, label in enumerate(self._data_labels):
                fmt = self._data_formats[j]
                reading = self._readings[label][i]
                self.table_ta.setItem(
                    i, j+2, _QTableWidgetItem(fmt.format(reading)))

        vbar = self.table_ta.verticalScrollBar()
        vbar.setValue(vbar.maximum())


class TemperatureTablePlotDialog(_QDialog, TablePlotWidget):
    """Temperature table and plot dialog."""

    _left_axis_1_label = 'Temperature [deg C]'
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (255, 225, 25),
        (70, 240, 240), (240, 50, 230), (170, 110, 40),
        (128, 0, 0), (0, 0, 0), (128, 128, 128), (0, 255, 0),
    ]

    def __init__(self, parent=None):
        _QDialog.__init__(self, parent)
        TablePlotWidget.__init__(self, parent)
        self.setWindowTitle('Temperature Readings')
        self.resize(1000, 800)
        self.setTableColumnSize(80)
        self.group_box.hide()
        self.tbt_autorange.hide()
        self.pbt_remove.hide()
        self.tbt_clear.hide()

    def accept(self):
        """Close dialog."""
        self.clear()
        self.closeDialogs()
        _QDialog.accept(self)

    def show(self, timestamp, readings):
        """Show dialog."""
        try:
            self._timestamp = timestamp
            self._readings = readings
            self._data_labels = list(self._readings.keys())
            self._data_formats = [
                self._left_axis_1_format]*len(self._data_labels)
            self._left_axis_1_data_labels = self._data_labels
            self.configurePlot()
            self.configureTable()
            self.updatePlot()
            self.updateTableValues()
            _QDialog.show(self)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class TemperatureTablePlotWidget(TablePlotWidget):
    """Temperature table and plot widget."""

    _left_axis_1_label = 'Temperature [deg C]'
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (255, 225, 25),
        (70, 240, 240), (240, 50, 230), (170, 110, 40),
        (128, 0, 0), (0, 0, 0), (128, 128, 128), (0, 255, 0),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Temperature Readings')
        self.resize(1000, 800)
        self.setTableColumnSize(80)
        self.group_box.hide()
        self.tbt_autorange.hide()
        self.pbt_remove.hide()
        self.tbt_clear.hide()

    def updateTemperatures(self, timestamp, readings):
        """Update temperature readings."""
        try:
            self._timestamp = timestamp
            self._readings = readings
            self._data_labels = list(self._readings.keys())
            self._data_formats = [
                self._left_axis_1_format]*len(self._data_labels)
            self._left_axis_1_data_labels = self._data_labels
            self.configurePlot()
            self.configureTable()
            self.updatePlot()
            self.updateTableValues()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class TemperatureChannelsWidget(_QWidget):
    """Temperature channels widget class."""

    channelChanged = _Signal()

    def __init__(self, channels, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)
        self.setWindowTitle("Temperature Channels")
        self.resize(275, 525)
        self.setFont(_font)

        main_layout = _QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout = _QGridLayout()

        group_box = _QGroupBox("Temperature [°C]")
        size_policy = _QSizePolicy(
            _QSizePolicy.Maximum, _QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)

        group_box.setSizePolicy(size_policy)
        group_box.setFont(_font_bold)
        group_box.setLayout(grid_layout)

        size_policy = _QSizePolicy(
            _QSizePolicy.Minimum, _QSizePolicy.Fixed)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        max_size = _QSize(155, 16777215)

        self.channels = channels
        for idx, ch in enumerate(self.channels):
            chb_label = "CH " + ch
            if ch == '101':
                chb_label = chb_label + ' (X)'
            elif ch == '102':
                chb_label = chb_label + ' (Y)'
            elif ch == '103':
                chb_label = chb_label + ' (Z)'
            chb = _QCheckBox(chb_label)
            chb.setFont(_font)
            chb.setChecked(False)
            chb.stateChanged.connect(self.clearChannelText)
            setattr(self, 'chb_channel' + ch, chb)

            le = _QLineEdit()
            le.setSizePolicy(size_policy)
            le.setMaximumSize(max_size)
            le.setFont(_font)
            le.setReadOnly(True)
            setattr(self, 'le_channel' + ch, le)

            grid_layout.addWidget(chb, idx, 0, 1, 1)
            grid_layout.addWidget(le, idx, 1, 1, 2)

        delay_label = _QLabel("Reading delay [s]:")
        size_policy = _QSizePolicy(
            _QSizePolicy.Preferred, _QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        delay_label.setSizePolicy(size_policy)
        delay_label.setFont(_font)
        delay_label.setAlignment(
            _Qt.AlignRight|_Qt.AlignTrailing|_Qt.AlignVCenter)
        grid_layout.addWidget(delay_label, len(self.channels)+1, 0, 1, 2)

        self.sbd_delay = _QDoubleSpinBox()
        size_policy = _QSizePolicy(_QSizePolicy.Maximum, _QSizePolicy.Fixed)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        self.sbd_delay.setSizePolicy(size_policy)
        self.sbd_delay.setFont(_font)
        self.sbd_delay.setValue(1.0)
        grid_layout.addWidget(self.sbd_delay, len(self.channels)+1, 2, 1, 1)

        main_layout.addWidget(group_box)
        self.setLayout(main_layout)

    @property
    def selected_channels(self):
        """Return the selected channels."""
        selected_channels = []
        for channel in self.channels:
            chb = getattr(self, 'chb_channel' + channel)
            if chb.isChecked():
                selected_channels.append(channel)
        return selected_channels

    @property
    def delay(self):
        """Return the reading delay."""
        return self.sbd_delay.value()

    def clearChannelText(self):
        """Clear channel text if channel is not selected."""
        for channel in self.channels:
            if channel not in self.selected_channels:
                le = getattr(self, 'le_channel' + channel)
                le.setText('')
        self.channelChanged.emit()

    def updateChannelText(self, channel, text):
        """Update channel text."""
        le = getattr(self, 'le_channel' + channel)
        le.setText(text)
