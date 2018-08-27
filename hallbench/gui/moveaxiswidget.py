# -*- coding: utf-8 -*-

"""Move axis widget for the Hall Bench Control application."""

from PyQt4.QtGui import (
    QWidget as _QWidget,
    QVBoxLayout as _QVBoxLayout,
    QMessageBox as _QMessageBox,
    )
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.currentpositionwidget import CurrentPositionWidget \
    as _CurrentPositionWidget


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

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.current_position_widget = _CurrentPositionWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.current_position_widget)
        self.ui.position_wg.setLayout(_layout)

        self.connectSignalSlots()

    @property
    def pmac(self):
        """Pmac object."""
        return self.window().devices.pmac

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            event.accept()
        except Exception:
            event.accept()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.targetvel_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.targetvel_le))
        self.ui.targetpos_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.targetpos_le))

        self.ui.selectaxis_cmb.currentIndexChanged.connect(
            self.updateVelocityAndPosition)

        self.ui.move_btn.clicked.connect(self.moveToTarget)
        self.ui.stop_btn.clicked.connect(self.stopAxis)

    def moveToTarget(self, axis):
        """Move axis to target position."""
        try:
            targetpos = float(self.ui.targetpos_le.text())
            targetvel = float(self.ui.targetvel_le.text())

            axis = self.selectedAxis()
            if axis is None:
                return

            velocity = self.pmac.get_velocity(axis)

            if targetvel != velocity:
                self.pmac.set_axis_speed(axis, targetvel)

            self.pmac.move_axis(axis, targetpos)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def selectedAxis(self):
        """Return the selected axis."""
        axis_str = self.ui.selectaxis_cmb.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in self.pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def setVelocityPositionStrFormat(self, obj):
        """Set the velocity and position string format."""
        try:
            value = float(obj.text())
            obj.setText(self._position_format.format(value))
        except Exception:
            self.updateVelocityAndPosition()

    def stopAxis(self):
        """Stop the selected axis."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return
            self.pmac.stop_axis(axis)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def updatePositions(self):
        """Update axes positions."""
        self.current_position_widget.updatePositions()

    def updateVelocityAndPosition(self):
        """Update velocity and position values for the selected axis."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return

            velocity = self.pmac.get_velocity(axis)
            position = self.pmac.get_position(axis)
            self.ui.targetvel_le.setText(self._position_format.format(
                velocity))
            self.ui.targetpos_le.setText(self._position_format.format(
                position))

            self.ui.targetvelunit_la.setText(self._axis_unit[axis] + '/s')
            self.ui.targetposunit_la.setText(self._axis_unit[axis])
        except Exception:
            pass
