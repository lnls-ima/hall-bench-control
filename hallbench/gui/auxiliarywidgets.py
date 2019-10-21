# -*- coding: utf-8 -*-

import sys as _sys
import numpy as _np
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
from hallbench.devices import pmac as _pmac


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
        self.view().pressed.connect(self.handle_item_pressed)
        self.setModel(_QStandardItemModel(self))

    def handle_item_pressed(self, index):
        """Change item check state."""
        item = self.model().itemFromIndex(index)
        if item.checkState() == _Qt.Checked:
            item.setCheckState(_Qt.Unchecked)
        else:
            item.setCheckState(_Qt.Checked)

    def checked_items(self):
        """Get checked items."""
        items = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                items.append(item)
        return items

    def checked_indexes(self):
        """Get checked indexes."""
        indexes = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                indexes.append(index)
        return indexes

    def checked_items_text(self):
        """Get checked items text."""
        items_text = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == _Qt.Checked:
                items_text.append(item.text())
        return items_text


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

        grid_layout = _QGridLayout()
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
                _Qt.AlignRight | _Qt.AlignTrailing | _Qt.AlignVCenter)
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
        self.timer.timeout.connect(self.update_positions)
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

    def update_positions(self):
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
            _QPixmap(_utils.get_icon_path('copy')), _QIcon.Normal, _QIcon.Off)

        sensors = ["X", "Y", "Z"]
        for idx, sensor in enumerate(sensors):
            vertical_layout = _QVBoxLayout()
            horizontal_layout = _QHBoxLayout()

            group_box = _QGroupBox("Sensor {0:s}".format(sensor.upper()))
            group_box.setFont(_font_bold)
            group_box.setAlignment(
                _Qt.AlignLeading | _Qt.AlignLeft | _Qt.AlignVCenter)
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
            setattr(self, 'tbl_sensor' + sensor.lower(), table)

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
            lambda: self.copy_to_clipboard('x'))
        self.pbt_sensory_copy.clicked.connect(
            lambda: self.copy_to_clipboard('y'))
        self.pbt_sensorz_copy.clicked.connect(
            lambda: self.copy_to_clipboard('z'))

        self.sb_sensorx_prec.valueChanged.connect(self.update_table_sensorx)
        self.sb_sensory_prec.valueChanged.connect(self.update_table_sensory)
        self.sb_sensorz_prec.valueChanged.connect(self.update_table_sensorz)

    def _update_table(self, table, data, precision):
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
        self.tbl_sensorx.clearContents()
        self.tbl_sensorx.setRowCount(0)
        self.tbl_sensory.clearContents()
        self.tbl_sensory.setRowCount(0)
        self.tbl_sensorz.clearContents()
        self.tbl_sensorz.setRowCount(0)

    def copy_to_clipboard(self, sensor):
        """Copy table data to clipboard."""
        try:
            table = getattr(self, 'tbl_sensor' + sensor)
            text = ""
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    text += str(table.item(r, c).text()) + "\t"
                text = text[:-1] + "\n"
            self.clip.setText(text)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def show(self, hall_probe):
        """Update hall probe object and show dialog."""
        self.local_hall_probe = hall_probe
        self.update_tables()
        super(InterpolationTableDialog, self).show()

    def update_tables(self):
        """Update table values."""
        if self.local_hall_probe is None:
            return
        self.update_table_sensorx()
        self.update_table_sensory()
        self.update_table_sensorz()

    def update_table_sensorx(self):
        """Update sensor x table values."""
        precision = self.sb_sensorx_prec.value()
        table = self.tbl_sensorx

        if self.local_hall_probe.sensorx is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorx.data
        self._update_table(table, data, precision)

    def update_table_sensory(self):
        """Update sensor y table values."""
        precision = self.sb_sensory_prec.value()
        table = self.tbl_sensory

        if self.local_hall_probe.sensory is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensory.data
        self._update_table(table, data, precision)

    def update_table_sensorz(self):
        """Update sensor z table values."""
        precision = self.sb_sensorz_prec.value()
        table = self.tbl_sensorz

        if self.local_hall_probe.sensorz is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorz.data
        self._update_table(table, data, precision)


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

        self.gb_move_axis = _QGroupBox("Move Axis")
        self.gb_move_axis.setFont(_font_bold)
        self.gb_move_axis.setLayout(grid_layout)

        label = _QLabel("Axis:")
        label.setFont(_font)
        grid_layout.addWidget(label, 0, 0, 1, 1)

        self.cmb_select_axis = _QComboBox()
        self.cmb_select_axis.setFont(_font)
        self.cmb_select_axis.addItem("")
        self.cmb_select_axis.addItem("#1 (+Z)")
        self.cmb_select_axis.addItem("#2 (+Y)")
        self.cmb_select_axis.addItem("#3 (+X)")
        self.cmb_select_axis.addItem("#5 (+A)")
        self.cmb_select_axis.addItem("#6 (+W)")
        self.cmb_select_axis.addItem("#7 (+V)")
        self.cmb_select_axis.addItem("#8 (+B)")
        self.cmb_select_axis.addItem("#9 (+C)")
        grid_layout.addWidget(self.cmb_select_axis, 0, 1, 1, 1)

        label = _QLabel("Velocity:")
        label.setFont(_font)
        grid_layout.addWidget(label, 1, 0, 1, 1)

        self.le_target_vel = _QLineEdit()
        self.le_target_vel.setSizePolicy(size_policy)
        self.le_target_vel.setFont(_font)
        grid_layout.addWidget(self.le_target_vel, 1, 1, 1, 1)

        self.la_target_vel_unit = _QLabel("")
        self.la_target_vel_unit.setFont(_font)
        grid_layout.addWidget(self.la_target_vel_unit, 1, 2, 1, 1)

        label = _QLabel("Position:")
        label.setFont(_font)
        grid_layout.addWidget(label, 2, 0, 1, 1)

        self.le_target_pos = _QLineEdit()
        self.le_target_pos.setSizePolicy(size_policy)
        self.le_target_pos.setFont(_font)
        grid_layout.addWidget(self.le_target_pos, 2, 1, 1, 1)

        self.la_target_pos_unit = _QLabel("")
        self.la_target_pos_unit.setFont(_font)
        grid_layout.addWidget(self.la_target_pos_unit, 2, 2, 1, 1)

        vertical_layout = _QVBoxLayout()
        vertical_layout.setSpacing(15)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.get_icon_path('move')), _QIcon.Normal, _QIcon.Off)
        self.pbt_move = _QPushButton("Move to Target")
        self.pbt_move.setIcon(icon)
        self.pbt_move.setMinimumSize(_QSize(200, 60))
        vertical_layout.addWidget(self.pbt_move)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.get_icon_path('stop')), _QIcon.Normal, _QIcon.Off)
        self.pbt_stop = _QPushButton("Stop Motor")
        self.pbt_stop.setIcon(icon)
        self.pbt_stop.setMinimumSize(_QSize(200, 60))
        self.pbt_stop.setObjectName("pbt_stop")
        vertical_layout.addWidget(self.pbt_stop)

        grid_layout.addLayout(vertical_layout, 3, 0, 1, 3)
        main_layout.addWidget(self.gb_move_axis)
        self.setLayout(main_layout)

        self.connect_signal_slots()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.le_target_vel.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(self.le_target_vel))
        self.le_target_pos.editingFinished.connect(
            lambda: self.set_velocity_position_str_format(self.le_target_pos))

        self.cmb_select_axis.currentIndexChanged.connect(
            self.update_velocity_and_position)

        self.pbt_move.clicked.connect(self.move_to_target)
        self.pbt_stop.clicked.connect(self.stop_axis)

    def move_to_target(self, axis):
        """Move axis to target position."""
        try:
            targetpos_str = self.le_target_pos.text()
            targetvel_str = self.le_target_vel.text()

            if len(targetpos_str) == 0 or len(targetvel_str) == 0:
                return

            targetpos = _utils.get_value_from_string(
                self.le_target_pos.text())
            targetvel = _utils.get_value_from_string(
                self.le_target_vel.text())

            axis = self.selected_axis()
            if axis is None:
                return

            velocity = _pmac.get_velocity(axis)

            if targetvel != velocity:
                _pmac.set_axis_speed(axis, targetvel)

            _pmac.move_axis(axis, targetpos)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to move axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def selected_axis(self):
        """Return the selected axis."""
        axis_str = self.cmb_select_axis.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in _pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def set_velocity_position_str_format(self, line_edit):
        """Set the velocity and position string format."""
        try:
            if not _utils.set_float_line_edit_text(line_edit, precision=3):
                self.update_velocity_and_position()
        except Exception:
            pass

    def stop_axis(self):
        """Stop the selected axis."""
        try:
            axis = self.selected_axis()
            if axis is None:
                return
            _pmac.stop_axis(axis)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def update_velocity_and_position(self):
        """Update velocity and position values for the selected axis."""
        try:
            axis = self.selected_axis()
            if axis is None:
                self.le_target_vel.setText('')
                self.le_target_pos.setText('')
                self.la_target_vel_unit.setText('')
                self.la_target_pos_unit.setText('')
                return

            velocity = _pmac.get_velocity(axis)
            position = _pmac.get_position(axis)
            self.le_target_vel.setText(self._position_format.format(
                velocity))
            self.le_target_pos.setText(self._position_format.format(
                position))

            self.la_target_vel_unit.setText(self._axis_unit[axis] + '/s')
            self.la_target_pos_unit.setText(self._axis_unit[axis])
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

    def update_plot(self):
        """Update plot."""
        self.canvas.draw()

    def show(self):
        """Show dialog."""
        self.update_plot()
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
            _QPixmap(_utils.get_icon_path('copy')), _QIcon.Normal, _QIcon.Off)

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
            setattr(self, 'tbl_sensor' + sensor.lower(), table)

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
            lambda: self.copy_to_clipboard('x'))
        self.pbt_sensory_copy.clicked.connect(
            lambda: self.copy_to_clipboard('y'))
        self.pbt_sensorz_copy.clicked.connect(
            lambda: self.copy_to_clipboard('z'))

        self.sb_sensorx_prec.valueChanged.connect(self.update_table_sensorx)
        self.sb_sensory_prec.valueChanged.connect(self.update_table_sensory)
        self.sb_sensorz_prec.valueChanged.connect(self.update_table_sensorz)

    def _update_table(self, table, data, precision):
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
        self.tbl_sensorx.clearContents()
        self.tbl_sensorx.setRowCount(0)
        self.tbl_sensory.clearContents()
        self.tbl_sensory.setRowCount(0)
        self.tbl_sensorz.clearContents()
        self.tbl_sensorz.setRowCount(0)

    def copy_to_clipboard(self, sensor):
        """Copy table data to clipboard."""
        try:
            table = getattr(self, 'tbl_sensor' + sensor)
            text = ""
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    text += str(table.item(r, c).text()) + "\t"
                text = text[:-1] + "\n"
            self.clip.setText(text)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def show(self, hall_probe=None):
        """Update hall probe object and show dialog."""
        self.local_hall_probe = hall_probe
        self.update_tables()
        super().show()

    def update_tables(self):
        """Update table values."""
        if self.local_hall_probe is None:
            return
        self.update_table_sensorx()
        self.update_table_sensory()
        self.update_table_sensorz()

    def update_table_sensorx(self):
        """Update sensor x table values."""
        precision = self.sb_sensorx_prec.value()
        table = self.tbl_sensorx
        if self.local_hall_probe.sensorx is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorx.data
        self._update_table(table, data, precision)

    def update_table_sensory(self):
        """Update sensor y table values."""
        precision = self.sb_sensory_prec.value()
        table = self.tbl_sensory
        if self.local_hall_probe.sensory is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensory.data
        self._update_table(table, data, precision)

    def update_table_sensorz(self):
        """Update sensor z table values."""
        precision = self.sb_sensorz_prec.value()
        table = self.tbl_sensorz
        if self.local_hall_probe.sensorz is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorz.data
        self._update_table(table, data, precision)


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

        self.pbt_apply.clicked.connect(self.tabs_preferences_changed)
        self.chb_connection.setChecked(True)
        self.chb_motors.setChecked(True)
        self.chb_measurement.setChecked(True)

    def tabs_preferences_changed(self):
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
        self.tbl_results = _QTableWidget()
        self.tbl_results.setAlternatingRowColors(True)
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.horizontalHeader().setDefaultSectionSize(120)

        self.pbt_copy = _QPushButton("Copy to clipboard")
        self.pbt_copy.clicked.connect(self.copy_to_clipboard)
        self.pbt_copy.setFont(_font_bold)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.tbl_results)
        _layout.addWidget(self.pbt_copy)
        self.setLayout(_layout)
        self.table_df = None

        self.resize(500, 200)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def add_items_to_table(self, text, i, j):
        """Add items to table."""
        item = _QTableWidgetItem(text)
        item.setFlags(_Qt.ItemIsSelectable | _Qt.ItemIsEnabled)
        self.tbl_results.setItem(i, j, item)

    def analyse_and_show_results(self):
        """Analyse data and add results to table."""
        self.tbl_results.clearContents()
        self.tbl_results.setRowCount(0)
        self.tbl_results.setColumnCount(0)

        if self.table_df is None:
            return

        self.tbl_results.setColumnCount(3)

        self.tbl_results.setHorizontalHeaderLabels(
            ['Mean', 'STD', 'Peak-Valey'])

        labels = [
            l for l in self.table_df.columns if l not in ['Date', 'Time']]

        self.tbl_results.setRowCount(len(labels))
        self.tbl_results.setVerticalHeaderLabels(labels)

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
            self.add_items_to_table('{0:.4f}'.format(mean), i, 0)
            self.add_items_to_table('{0:.4f}'.format(std), i, 1)
            self.add_items_to_table('{0:.4f}'.format(peak_valey), i, 2)

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def clear(self):
        """Clear data and table."""
        self.table_df = None
        self.tbl_results.clearContents()
        self.tbl_results.setRowCount(0)
        self.tbl_results.setColumnCount(0)

    def copy_to_clipboard(self):
        """Copy table data to clipboard."""
        df = _utils.table_to_data_frame(self.tbl_results)
        if df is not None:
            df.to_clipboard(excel=True)

    def show(self, table_df):
        """Show dialog."""
        self.table_df = table_df
        self.analyse_and_show_results()
        super().show()

    def update_data(self, table_df):
        """Update table data."""
        self.table_df = table_df
        self.analyse_and_show_results()


class TableDialog(_QDialog):
    """Table dialog class."""

    def __init__(self, parent=None):
        """Add table widget and copy button."""
        super().__init__(parent)
        self.resize(800, 500)
        self.setFont(_font)

        self.setWindowTitle("Data Table")
        self.tbl_data = _QTableWidget()
        self.tbl_data.setAlternatingRowColors(True)
        self.tbl_data.verticalHeader().hide()
        self.tbl_data.horizontalHeader().setStretchLastSection(True)
        self.tbl_data.horizontalHeader().setDefaultSectionSize(120)

        self.pbt_copy = _QPushButton("Copy to clipboard")
        self.pbt_copy.clicked.connect(self.copy_to_clipboard)
        self.pbt_copy.setFont(_font_bold)

        _layout = _QVBoxLayout()
        _layout.addWidget(self.tbl_data)
        _layout.addWidget(self.pbt_copy)
        self.setLayout(_layout)
        self.table_df = None

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def add_items_to_table(self, text, i, j):
        """Add items to table."""
        item = _QTableWidgetItem(text)
        item.setFlags(_Qt.ItemIsSelectable | _Qt.ItemIsEnabled)
        self.tbl_data.setItem(i, j, item)

    def clear(self):
        """Clear data and table."""
        self.table_df = None
        self.tbl_data.clearContents()
        self.tbl_data.setRowCount(0)
        self.tbl_data.setColumnCount(0)

    def copy_to_clipboard(self):
        """Copy table data to clipboard."""
        df = _utils.table_to_data_frame(self.tbl_data)
        if df is not None:
            df.to_clipboard(excel=True)

    def show(self, table_df):
        """Show dialog."""
        self.table_df = table_df
        self.update_table()
        super().show()

    def update_data(self, table_df):
        """Update table data."""
        self.table_df = table_df
        self.update_table()

    def update_table(self):
        """Add data to table."""
        self.tbl_data.clearContents()
        self.tbl_data.setRowCount(0)
        self.tbl_data.setColumnCount(0)

        if self.table_df is None:
            return

        nrows = self.table_df.shape[0]
        ncols = self.table_df.shape[1]

        self.tbl_data.setRowCount(nrows)
        self.tbl_data.setColumnCount(ncols)

        columns = self.table_df.columns.values
        self.tbl_data.setHorizontalHeaderLabels(columns)

        for i in range(nrows):
            for j in range(ncols):
                if columns[j] == 'ID':
                    text = '{0:d}'.format(int(self.table_df.iloc[i, j]))
                else:
                    text = str(self.table_df.iloc[i, j])
                self.add_items_to_table(text, i, j)


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
        self.add_widgets()
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
        self.update_monitor_interval()
        self.timer.timeout.connect(lambda: self.read_value(monitor=True))

        # create table analysis dialog
        self.table_analysis_dialog = TableAnalysisDialog()

        # add legend to plot
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.pw_plot.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.right_axis_1 = None
        self.right_axis_2 = None
        self.configure_plot()
        self.configure_table()
        self.connect_signal_slots()

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            self.close_dialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def add_last_value_to_table(self):
        """Add the last value read to table."""
        if len(self._timestamp) == 0:
            return

        n = self.tbl_table.rowCount() + 1
        self.tbl_table.setRowCount(n)

        dt = _datetime.datetime.fromtimestamp(self._timestamp[-1])
        date = dt.strftime("%d/%m/%Y")
        hour = dt.strftime("%H:%M:%S")
        self.tbl_table.setItem(n-1, 0, _QTableWidgetItem(date))
        self.tbl_table.setItem(n-1, 1, _QTableWidgetItem(hour))

        for j, label in enumerate(self._data_labels):
            fmt = self._data_formats[j]
            reading = self._readings[label][-1]
            self.tbl_table.setItem(
                n-1, j+2, _QTableWidgetItem(fmt.format(reading)))

        vbar = self.tbl_table.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def add_widgets(self):
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
        label.setAlignment(
            _Qt.AlignRight | _Qt.AlignTrailing | _Qt.AlignVCenter)
        self.horizontal_layout_3.addWidget(label)

        self.sbd_monitor_step = _QDoubleSpinBox()
        self.sbd_monitor_step.setDecimals(1)
        self.sbd_monitor_step.setMinimum(0.1)
        self.sbd_monitor_step.setMaximum(60.0)
        self.sbd_monitor_step.setProperty("value", 10.0)
        self.horizontal_layout_3.addWidget(self.sbd_monitor_step)

        self.cmb_monitor_unit = _QComboBox()
        self.cmb_monitor_unit.addItem("sec")
        self.cmb_monitor_unit.addItem("min")
        self.cmb_monitor_unit.addItem("hour")
        self.horizontal_layout_3.addWidget(self.cmb_monitor_unit)
        self.vertical_layout_2.addLayout(self.horizontal_layout_3)

        # Group box with read and monitor buttons
        self.group_box = _QGroupBox()
        self.group_box.setMinimumSize(_QSize(270, 0))
        self.group_box.setTitle("")
        self.group_box.setLayout(self.vertical_layout_2)

        # Table widget
        self.tbl_table = _QTableWidget()
        sizePolicy = _QSizePolicy(
            _QSizePolicy.Expanding, _QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.tbl_table.sizePolicy().hasHeightForWidth())
        self.tbl_table.setSizePolicy(sizePolicy)
        self.tbl_table.setVerticalScrollBarPolicy(_Qt.ScrollBarAlwaysOn)
        self.tbl_table.setHorizontalScrollBarPolicy(_Qt.ScrollBarAsNeeded)
        self.tbl_table.setEditTriggers(_QAbstractItemView.NoEditTriggers)
        self.tbl_table.setColumnCount(0)
        self.tbl_table.setRowCount(0)
        self.tbl_table.horizontalHeader().setVisible(True)
        self.tbl_table.horizontalHeader().setCascadingSectionResizes(False)
        self.tbl_table.horizontalHeader().setDefaultSectionSize(200)
        self.tbl_table.horizontalHeader().setHighlightSections(True)
        self.tbl_table.horizontalHeader().setMinimumSectionSize(80)
        self.tbl_table.horizontalHeader().setStretchLastSection(True)
        self.tbl_table.verticalHeader().setVisible(False)
        self.horizontal_layout_4.addWidget(self.tbl_table)

        # Tool buttons
        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.get_icon_path('font')), _QIcon.Normal, _QIcon.Off)
        self.tbt_autorange = _QToolButton()
        self.tbt_autorange.setIcon(icon)
        self.tbt_autorange.setIconSize(icon_size)
        self.tbt_autorange.setCheckable(True)
        self.tbt_autorange.setToolTip('Turn on plot autorange.')
        self.vertical_layout_3.addWidget(self.tbt_autorange)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.get_icon_path('save')), _QIcon.Normal, _QIcon.Off)
        self.tbt_save = _QToolButton()
        self.tbt_save.setIcon(icon)
        self.tbt_save.setIconSize(icon_size)
        self.tbt_save.setToolTip('Save table data to file.')
        self.vertical_layout_3.addWidget(self.tbt_save)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.get_icon_path('copy')), _QIcon.Normal, _QIcon.Off)
        self.tbt_copy = _QToolButton()
        self.tbt_copy.setIcon(icon)
        self.tbt_copy.setIconSize(icon_size)
        self.tbt_copy.setToolTip('Copy table data.')
        self.vertical_layout_3.addWidget(self.tbt_copy)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.get_icon_path('stats')), _QIcon.Normal, _QIcon.Off)
        self.pbt_stats = _QToolButton()
        self.pbt_stats.setIcon(icon)
        self.pbt_stats.setIconSize(icon_size)
        self.pbt_stats.setToolTip('Show data statistics.')
        self.vertical_layout_3.addWidget(self.pbt_stats)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(
                _utils.get_icon_path('delete')), _QIcon.Normal, _QIcon.Off)
        self.pbt_remove = _QToolButton()
        self.pbt_remove.setIcon(icon)
        self.pbt_remove.setIconSize(icon_size)
        self.pbt_remove.setToolTip('Remove selected lines from table.')
        self.vertical_layout_3.addWidget(self.pbt_remove)

        icon = _QIcon()
        icon.addPixmap(
            _QPixmap(_utils.get_icon_path('clear')), _QIcon.Normal, _QIcon.Off)
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

    def add_widgets_next_to_plot(self, widget_list):
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

    def add_widgets_next_to_table(self, widget_list):
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

    def clear_legend_items(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def clear_button_clicked(self):
        """Clear all values."""
        if len(self._timestamp) == 0:
            return

        msg = 'Clear table data?'
        reply = _QMessageBox.question(
            self, 'Message', msg, buttons=_QMessageBox.No | _QMessageBox.Yes,
            defaultButton=_QMessageBox.No)

        if reply == _QMessageBox.Yes:
            self.clear()

    def clear(self):
        """Clear all values."""
        self._timestamp = []
        for label in self._data_labels:
            self._readings[label] = []
        self.update_table_values()
        self.update_plot()
        self.update_table_analysis_dialog()

    def close_dialogs(self):
        """Close dialogs."""
        try:
            self.table_analysis_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def configure_plot(self):
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
            self.right_axis_1 = _utils.plot_item_add_first_right_axis(
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
            self.right_axis_2 = _utils.plot_item_add_second_right_axis(
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
        self.update_legend_items()

    def configure_table(self):
        """Configure table."""
        col_labels = ['Date', 'Time']
        for label in self._data_labels:
            col_labels.append(label)
        self.tbl_table.setColumnCount(len(col_labels))
        self.tbl_table.setHorizontalHeaderLabels(col_labels)
        self.tbl_table.setAlternatingRowColors(True)

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.pbt_read.clicked.connect(lambda: self.read_value(monitor=False))
        self.pbt_monitor.toggled.connect(self.monitor_value)
        self.sbd_monitor_step.valueChanged.connect(
            self.update_monitor_interval)
        self.cmb_monitor_unit.currentIndexChanged.connect(
            self.update_monitor_interval)
        self.tbt_autorange.toggled.connect(self.enable_autorange)
        self.tbt_save.clicked.connect(self.save_to_file)
        self.tbt_copy.clicked.connect(self.copy_to_clipboard)
        self.pbt_stats.clicked.connect(self.show_table_analysis_dialog)
        self.pbt_remove.clicked.connect(self.remove_value)
        self.tbt_clear.clicked.connect(self.clear_button_clicked)

    def copy_to_clipboard(self):
        """Copy table data to clipboard."""
        df = _utils.table_to_data_frame(self.tbl_table)
        if df is not None:
            df.to_clipboard(excel=True)

    def enable_autorange(self, checked):
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

    def hide_right_axes(self):
        """Hide right axes."""
        if self.right_axis_1 is not None:
            self.right_axis_1.setStyle(showValues=False)
            self.right_axis_1.setLabel('')
        if self.right_axis_2 is not None:
            self.right_axis_2.setStyle(showValues=False)
            self.right_axis_2.setLabel('')

    def monitor_value(self, checked):
        """Monitor values."""
        if checked:
            self.pbt_read.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.pbt_read.setEnabled(True)

    def read_value(self, monitor=False):
        """Read value."""
        pass

    def remove_value(self):
        """Remove value from list."""
        selected = self.tbl_table.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self._timestamp)

        self._timestamp = [
            self._timestamp[i] for i in range(n) if i not in rows]

        for label in self._data_labels:
            readings = self._readings[label]
            self._readings[label] = [
                readings[i] for i in range(n) if i not in rows]

        self.update_table_values()
        self.update_plot()
        self.update_table_analysis_dialog()

    def save_to_file(self):
        """Save table values to file."""
        df = _utils.table_to_data_frame(self.tbl_table)
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

    def set_table_column_size(self, size):
        """Set table horizontal header default section size."""
        self.tbl_table.horizontalHeader().setDefaultSectionSize(size)

    def show_table_analysis_dialog(self):
        """Show table analysis dialog."""
        df = _utils.table_to_data_frame(self.tbl_table)
        self.table_analysis_dialog.accept()
        self.table_analysis_dialog.show(df)

    def update_legend_items(self):
        """Update legend items."""
        self.clear_legend_items()
        self._legend_items = []
        for label in self._data_labels:
            legend_label = label.split('[')[0]
            self._legend_items.append(legend_label)
            self.legend.addItem(self._graphs[label], legend_label)

    def update_monitor_interval(self):
        """Update monitor interval value."""
        index = self.cmb_monitor_unit.currentIndex()
        if index == 0:
            mf = 1000
        elif index == 1:
            mf = 1000*60
        else:
            mf = 1000*3600
        self.timer.setInterval(self.sbd_monitor_step.value()*mf)

    def update_plot(self):
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

    def update_table_analysis_dialog(self):
        """Update table analysis dialog."""
        self.table_analysis_dialog.update_data(
            _utils.table_to_data_frame(self.tbl_table))

    def update_table_values(self):
        """Update table values."""
        n = len(self._timestamp)
        self.tbl_table.clearContents()
        self.tbl_table.setRowCount(n)

        for i in range(n):
            dt = _datetime.datetime.fromtimestamp(self._timestamp[i])
            date = dt.strftime("%d/%m/%Y")
            hour = dt.strftime("%H:%M:%S")
            self.tbl_table.setItem(i, 0, _QTableWidgetItem(date))
            self.tbl_table.setItem(i, 1, _QTableWidgetItem(hour))

            for j, label in enumerate(self._data_labels):
                fmt = self._data_formats[j]
                reading = self._readings[label][i]
                self.tbl_table.setItem(
                    i, j+2, _QTableWidgetItem(fmt.format(reading)))

        vbar = self.tbl_table.verticalScrollBar()
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
        self.set_table_column_size(80)
        self.group_box.hide()
        self.tbt_autorange.hide()
        self.pbt_remove.hide()
        self.tbt_clear.hide()

    def accept(self):
        """Close dialog."""
        self.clear()
        self.close_dialogs()
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
            self.configure_plot()
            self.configure_table()
            self.update_plot()
            self.update_table_values()
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
        self.set_table_column_size(80)
        self.group_box.hide()
        self.tbt_autorange.hide()
        self.pbt_remove.hide()
        self.tbt_clear.hide()

    def update_temperatures(self, timestamp, readings):
        """Update temperature readings."""
        try:
            self._timestamp = timestamp
            self._readings = readings
            self._data_labels = list(self._readings.keys())
            self._data_formats = [
                self._left_axis_1_format]*len(self._data_labels)
            self._left_axis_1_data_labels = self._data_labels
            self.configure_plot()
            self.configure_table()
            self.update_plot()
            self.update_table_values()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class TemperatureChannelsWidget(_QWidget):
    """Temperature channels widget class."""

    channel_changed = _Signal()

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
            chb.stateChanged.connect(self.clear_channel_text)
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
            _Qt.AlignRight | _Qt.AlignTrailing | _Qt.AlignVCenter)
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

    def clear_channel_text(self):
        """Clear channel text if channel is not selected."""
        for channel in self.channels:
            if channel not in self.selected_channels:
                le = getattr(self, 'le_channel' + channel)
                le.setText('')
        self.channel_changed.emit()

    def update_channel_text(self, channel, text):
        """Update channel text."""
        le = getattr(self, 'le_channel' + channel)
        le.setText(text)
