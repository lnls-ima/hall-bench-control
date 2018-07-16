# -*- coding: utf-8 -*-

"""Motors widget for the Hall Bench Control application."""

import time as _time
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.positionwidget import PositionWidget as _PositionWidget


class MotorsWidget(_QWidget):
    """Motors Widget class for the Hall Bench Control application."""

    _align_bench_time_interval = 0.5  # [s]
    _axis_unit = {
        1: 'mm', 2: 'mm', 3: 'mm', 5: 'deg',
        6: 'mm', 7: 'mm', 8: 'deg', 9: 'deg',
    }

    def __init__(self, parent=None):
        """Set up the ui, add position widget and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.position_widget = _PositionWidget(self)
        self.ui.position_lt.addWidget(self.position_widget)

        # disable combo box itens
        for item in range(self.ui.selectaxis_cmb.count()):
            self.ui.selectaxis_cmb.model().item(item).setEnabled(False)

        # create connections
        self.ui.minax1_le.editingFinished.connect(
            lambda: self.setAxisLimitsStrFormat(self.ui.minax1_le))
        self.ui.minax2_le.editingFinished.connect(
            lambda: self.setAxisLimitsStrFormat(self.ui.minax2_le))
        self.ui.minax3_le.editingFinished.connect(
            lambda: self.setAxisLimitsStrFormat(self.ui.minax3_le))
        self.ui.maxax1_le.editingFinished.connect(
            lambda: self.setAxisLimitsStrFormat(self.ui.maxax1_le))
        self.ui.maxax2_le.editingFinished.connect(
            lambda: self.setAxisLimitsStrFormat(self.ui.maxax2_le))
        self.ui.maxax3_le.editingFinished.connect(
            lambda: self.setAxisLimitsStrFormat(self.ui.maxax3_le))

        self.ui.targetvel_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.targetvel_le))
        self.ui.reldisp_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.reldisp_le))
        self.ui.targetpos_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.targetpos_le))

        self.ui.reldisp_le.editingFinished.connect(self.updateTargetPos)
        self.ui.targetpos_le.editingFinished.connect(self.updateRelDisp)

        self.ui.selectaxis_cmb.currentIndexChanged.connect(
            self.updateVelocityAndPosition)

        self.ui.activate_btn.clicked.connect(self.activateBench)
        self.ui.stopall_btn.clicked.connect(self.stopAllAxis)
        self.ui.killall_btn.clicked.connect(self.killAllAxis)
        self.ui.homming_btn.clicked.connect(self.startHomming)
        self.ui.setlimits_btn.clicked.connect(self.setAxisLimits)
        self.ui.resetlimits_btn.clicked.connect(self.resetAxisLimits)
        self.ui.move_btn.clicked.connect(self.moveToTarget)
        self.ui.stop_btn.clicked.connect(self.stopAxis)

    @property
    def pmac(self):
        """Pmac object."""
        return self.window().devices.pmac

    def activateBench(self):
        """Activate the bench and enable control."""
        if self.pmac is None:
            return

        if self.pmac.activate_bench():
            self.setHommingEnabled(True)
            self.setAxisLimitsEnabled(True)
            self.releaseAccessToMovement()
        else:
            self.setHommingEnabled(False)
            self.setAxisLimitsEnabled(False)
            self.setMovementEnabled(False)
            message = 'Failed to activate bench.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

    def killAllAxis(self):
        """Kill all axis."""
        if self.pmac is None:
            return
        self.pmac.kill_all_axis()

    def moveToTarget(self, axis):
        """Move Hall probe to target position."""
        axis = self.selectedAxis()
        if axis is None:
            return

        targetvel = float(self.ui.targetvel_le.text())
        targetpos = float(self.ui.targetpos_le.text())
        velocity = self.pmac.get_velocity(axis)

        if targetvel != velocity:
            self.pmac.set_axis_speed(axis, targetvel)

        self.pmac.move_axis(axis, targetpos)

    def releaseAccessToMovement(self):
        """Check homming status and enable movement."""
        if self.pmac is None:
            return

        list_of_axis = self.pmac.commands.list_of_axis

        item = 0
        hommming_status = []
        for axis in list_of_axis:
            if (self.pmac.axis_status(axis) & 1024) != 0:
                self.ui.selectaxis_cmb.model().item(item).setEnabled(True)
                hommming_status.append(True)
            else:
                self.ui.selectaxis_cmb.model().item(item).setEnabled(False)
                hommming_status.append(False)
            item += 1

        if any(hommming_status):
            self.setMovementEnabled(True)
            self.updateVelocityAndPosition()
        else:
            self.setMovementEnabled(False)

        if all(hommming_status):
            self.updateMainTabStatus(3, True)
        else:
            self.updateMainTabStatus(3, False)

    def resetAxisLimits(self):
        """Reset axis limits."""
        if self.pmac is None:
            return

        neg_list = self.pmac.commands.i_softlimit_neg_list
        pos_list = self.pmac.commands.i_softlimit_pos_list

        if self.pmac.get_response(self.pmac.set_par(neg_list[0], 0)):
            self.ui.minax1_le.setText('')

        if self.pmac.get_response(self.pmac.set_par(pos_list[0], 0)):
            self.ui.maxax1_le.setText('')

        if self.pmac.get_response(self.pmac.set_par(neg_list[1], 0)):
            self.ui.minax2_le.setText('')

        if self.pmac.get_response(self.pmac.set_par(pos_list[1], 0)):
            self.ui.maxax2_le.setText('')

        if self.pmac.get_response(self.pmac.set_par(neg_list[2], 0)):
            self.ui.minax3_le.setText('')

        if self.pmac.get_response(self.pmac.set_par(pos_list[2], 0)):
            self.ui.maxax3_le.setText('')

    def selectedAxis(self):
        """Return the selected axis."""
        if self.pmac is None:
            return None

        axis_str = self.ui.selectaxis_cmb.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in self.pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def setAxisLimits(self):
        """Set axis limits."""
        if self.pmac is None:
            return

        neg_list = self.pmac.commands.i_softlimit_neg_list
        pos_list = self.pmac.commands.i_softlimit_pos_list
        cts_mm_axis = self.pmac.commands.CTS_MM_AXIS

        minax1 = float(self.ui.minax1_le.text())*cts_mm_axis[0]
        maxax1 = float(self.ui.maxax1_le.text())*cts_mm_axis[0]
        self.pmac.get_response(self.pmac.set_par(neg_list[0], minax1))
        self.pmac.get_response(self.pmac.set_par(pos_list[0], maxax1))

        minax2 = float(self.ui.minax1_le.text())*cts_mm_axis[1]
        maxax2 = float(self.ui.maxax1_le.text())*cts_mm_axis[1]
        self.pmac.get_response(self.pmac.set_par(neg_list[1], minax2))
        self.pmac.get_response(self.pmac.set_par(pos_list[1], maxax2))

        minax3 = float(self.ui.minax1_le.text())*cts_mm_axis[2]
        maxax3 = float(self.ui.maxax1_le.text())*cts_mm_axis[2]
        self.pmac.get_response(self.pmac.set_par(neg_list[2], minax3))
        self.pmac.get_response(self.pmac.set_par(pos_list[2], maxax3))

    def setAxisLimitsEnabled(self, enabled):
        """Enable/Disable axis limits controls."""
        self.ui.limits_gb.setEnabled(enabled)
        self.ui.setlimits_btn.setEnabled(enabled)
        self.ui.resetlimits_btn.setEnabled(enabled)

    def setAxisLimitsStrFormat(self, obj):
        """Set the axis limit string format."""
        try:
            value = float(obj.text())
            obj.setText('{0:0.4f}'.format(value))
        except Exception:
            obj.setText('')

    def setHommingEnabled(self, enabled):
        """Enable/Disable homming controls."""
        self.ui.homming_gb.setEnabled(enabled)
        self.ui.homming_btn.setEnabled(enabled)

    def setMovementEnabled(self, enabled):
        """Enable/Disable movement controls."""
        self.ui.moveaxis_gb.setEnabled(enabled)
        self.ui.move_btn.setEnabled(enabled)
        self.ui.stop_btn.setEnabled(enabled)

    def setVelocityPositionStrFormat(self, obj):
        """Set the velocity and position string format."""
        try:
            value = float(obj.text())
            obj.setText('{0:0.4f}'.format(value))
        except Exception:
            self.updateVelocityAndPosition()

    def startHomming(self):
        """Homming of the selected axes."""
        if self.pmac is None:
            return

        axis_homming_mask = 0
        list_of_axis = self.pmac.commands.list_of_axis

        for axis in list_of_axis:
            obj = getattr(self.ui, 'hommingax' + str(axis) + '_chb')
            val = int(obj.isChecked())
            axis_homming_mask += (val << (axis-1))

        self.pmac.align_bench(axis_homming_mask)
        _time.sleep(self._align_bench_time_interval)

        while int(self.pmac.read_response(
                self.pmac.commands.prog_running)) == 1:
            _time.sleep(self._align_bench_time_interval)
        else:
            self.releaseAccessToMovement()
            message = 'Finished homming of the selected axes.'
            _QMessageBox.information(
                self, 'Hommming', message, _QMessageBox.Ok)

    def stopAllAxis(self):
        """Stop all axis."""
        if self.pmac is None:
            return

        self.pmac.stop_all_axis()

    def stopAxis(self):
        """Stop the selected axis."""
        axis = self.selectedAxis()
        if axis is None:
            return
        self.pmac.stop_axis(axis)

    def updatePositions(self):
        """Update axes positions."""
        if self.pmac is None:
            return
        self.position_widget.updatePositions()

    def updateRelDisp(self):
        """Update relative displacement value."""
        axis = self.selectedAxis()
        if axis is None:
            return

        position = self.pmac.get_position(axis)
        targetpos = float(self.ui.targetpos_le.text())
        reldisp = targetpos - position
        self.ui.reldisp_le.setText('{0:0.4f}'.format(reldisp))

    def updateTargetPos(self):
        """Update target position value."""
        axis = self.selectedAxis()
        if axis is None:
            return

        position = self.pmac.get_position(axis)
        reldisp = float(self.ui.reldisp_le.text())
        targetpos = position + reldisp
        self.ui.targetpos_le.setText('{0:0.4f}'.format(targetpos))

    def updateVelocityAndPosition(self):
        """Update velocity and position values for the selected axis."""
        axis = self.selectedAxis()
        if axis is None:
            return

        velocity = self.pmac.get_velocity(axis)
        position = self.pmac.get_position(axis)
        self.ui.targetvel_le.setText('{0:0.4f}'.format(velocity))
        self.ui.reldisp_le.setText('{0:0.4f}'.format(0))
        self.ui.targetpos_le.setText('{0:0.4f}'.format(position))

        self.ui.targetvelunit_la.setText(self._axis_unit[axis] + '/s')
        self.ui.reldispunit_la.setText(self._axis_unit[axis])
        self.ui.targetposunit_la.setText(self._axis_unit[axis])
