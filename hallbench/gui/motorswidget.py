# -*- coding: utf-8 -*-

"""Motors widget for the Hall Bench Control application."""

import sys as _sys
import time as _time
import numpy as _np
import traceback as _traceback
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
import qtpy.uic as _uic

from hallbench.gui import utils as _utils
from hallbench.gui.auxiliarywidgets import CurrentPositionWidget \
    as _CurrentPositionWidget


class MotorsWidget(_QWidget):
    """Motors Widget class for the Hall Bench Control application."""

    _align_bench_time_interval = 0.5  # [s]
    _axis_unit = {
        1: 'mm', 2: 'mm', 3: 'mm', 5: 'deg',
        6: 'mm', 7: 'mm', 8: 'deg', 9: 'deg',
    }
    _position_format = '{0:0.3f}'

    def __init__(self, parent=None):
        """Set up the ui, add position widget and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.current_position_widget = _CurrentPositionWidget(self)
        self.ui.position_lt.addWidget(self.current_position_widget)

        # variables initialization
        self.homing = False
        self.stop_trigger = False

        # disable combo box itens
        for item in range(self.ui.cmb_selectaxis.count()):
            self.ui.cmb_selectaxis.model().item(item).setEnabled(False)
        for item in range(self.ui.cmb_selecttrigaxis.count()):
            self.ui.cmb_selecttrigaxis.model().item(item).setEnabled(False)

        # disable trigger delay
        self.ui.chb_trigpause.setChecked(False)
        self.ui.sbd_trigdelay.setEnabled(False)

        self.connectSignalSlots()

    @property
    def pmac(self):
        """Pmac communication class."""
        return _QApplication.instance().devices.pmac

    def activateBench(self):
        """Activate the bench and enable control."""
        try:
            if self.pmac.activate_bench():
                self.setHomingEnabled(True)
                self.setAxisLimitsEnabled(True)
                self.ui.pbt_setlimits.setEnabled(False)
                self.releaseAccessToMovement()

            else:
                self.setHomingEnabled(False)
                self.setAxisLimitsEnabled(False)
                self.setMovementEnabled(False)
                self.setTriggerEnabled(False)
                msg = 'Failed to activate bench.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to activate bench.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

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
        self.ui.le_minax1.editingFinished.connect(
            lambda: _utils.setFloatLineEditText(
                self.ui.le_minax1, precision=3))
        self.ui.le_minax2.editingFinished.connect(
            lambda: _utils.setFloatLineEditText(
                self.ui.le_minax2, precision=3))
        self.ui.le_minax3.editingFinished.connect(
            lambda: _utils.setFloatLineEditText(
                self.ui.le_minax3, precision=3))
        self.ui.le_maxax1.editingFinished.connect(
            lambda: _utils.setFloatLineEditText(
                self.ui.le_maxax1, precision=3))
        self.ui.le_maxax2.editingFinished.connect(
            lambda: _utils.setFloatLineEditText(
                self.ui.le_maxax2, precision=3))
        self.ui.le_maxax3.editingFinished.connect(
            lambda: _utils.setFloatLineEditText(
                self.ui.le_maxax3, precision=3))

        self.ui.le_targetvel.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.le_targetvel))
        self.ui.le_reldisp.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.le_reldisp))
        self.ui.le_targetpos.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.le_targetpos))

        self.ui.le_trigvel.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.le_trigvel))
        self.ui.le_trigstart.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.le_trigstart))
        self.ui.le_trigstep.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.le_trigstep))
        self.ui.le_trigend.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.le_trigend))

        self.ui.le_trigstart.editingFinished.connect(self.fixPositionValues)
        self.ui.le_trigstep.editingFinished.connect(self.fixPositionValues)
        self.ui.le_trigend.editingFinished.connect(self.fixPositionValues)

        self.ui.le_reldisp.editingFinished.connect(self.updateTargetPos)
        self.ui.le_targetpos.editingFinished.connect(self.updateRelDisp)

        self.ui.cmb_selectaxis.currentIndexChanged.connect(
            self.updateVelocityAndPosition)

        self.ui.cmb_selecttrigaxis.currentIndexChanged.connect(
            self.updateTrigAxisVelocity)
        self.ui.chb_trigpause.stateChanged.connect(self.enableTriggerDelay)

        self.ui.le_minax1.editingFinished.connect(self.enableSetLimitsButton)
        self.ui.le_maxax1.editingFinished.connect(self.enableSetLimitsButton)
        self.ui.le_minax2.editingFinished.connect(self.enableSetLimitsButton)
        self.ui.le_maxax2.editingFinished.connect(self.enableSetLimitsButton)
        self.ui.le_minax3.editingFinished.connect(self.enableSetLimitsButton)
        self.ui.le_maxax3.editingFinished.connect(self.enableSetLimitsButton)

        self.ui.pbt_activate.clicked.connect(self.activateBench)
        self.ui.pbt_stopall.clicked.connect(self.stopAllAxis)
        self.ui.pbt_killall.clicked.connect(self.killAllAxis)
        self.ui.pbt_homing.clicked.connect(self.startHoming)
        self.ui.pbt_setlimits.clicked.connect(self.setAxisLimits)
        self.ui.pbt_resetlimits.clicked.connect(self.resetAxisLimits)
        self.ui.pbt_move.clicked.connect(self.moveToTarget)
        self.ui.pbt_stop.clicked.connect(self.stopAxis)
        self.ui.pbt_trigandmove.clicked.connect(self.setTriggerandMove)
        self.ui.pbt_trigstop.clicked.connect(self.stopTriggerAxis)

    def enableSetLimitsButton(self):
        """Enable set limits button."""
        self.ui.pbt_setlimits.setEnabled(True)

    def enableTriggerDelay(self):
        """Enable or disable trigger delay."""
        if self.ui.chb_trigpause.isChecked():
            self.ui.sbd_trigdelay.setEnabled(True)
        else:
            self.ui.sbd_trigdelay.setEnabled(False)

    def fixPositionValues(self):
        """Fix step and end position value."""
        start = _utils.getValueFromStringExpresssion(
            self.ui.le_trigstart.text())
        if start is None:
            return

        step = _utils.getValueFromStringExpresssion(self.ui.le_trigstep.text())
        if step is None:
            return

        end = _utils.getValueFromStringExpresssion(self.ui.le_trigend.text())
        if end is None:
            return

        if step == 0:
            self.ui.le_trigend.setText('{0:0.4f}'.format(start))
            return

        npts = _np.abs(_np.round(round((end - start) / step, 4) + 1))
        if start <= end and step < 0:
            self.ui.le_trigstep.setText('')
            return
        elif start > end and step > 0:
            self.ui.le_trigstep.setText('')
            return
        elif start <= end:
            corrected_step = _np.abs(step)
            corrected_end = start + (npts-1)*corrected_step
        else:
            corrected_step = _np.abs(step)*(-1)
            corrected_end = start + (npts-1)*corrected_step

        self.ui.le_trigstep.setText('{0:0.4f}'.format(corrected_step))
        self.ui.le_trigend.setText('{0:0.4f}'.format(corrected_end))

    def killAllAxis(self):
        """Kill all axis."""
        try:
            self.pmac.kill_all_axis()
            self.setHomingEnabled(False)
            self.setAxisLimitsEnabled(False)
            self.setMovementEnabled(False)
            self.setTriggerEnabled(False)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to kill axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def moveToTarget(self, axis):
        """Move Hall probe to target position."""
        try:
            targetpos = _utils.getValueFromStringExpresssion(
                self.ui.le_targetpos.text())
            targetvel = _utils.getValueFromStringExpresssion(
                self.ui.le_targetvel.text())
            if targetpos is None or targetvel is None:
                return

            axis = self.selectedAxis()
            if axis is None:
                return

            velocity = self.pmac.get_velocity(axis)

            if targetvel != velocity:
                self.pmac.set_axis_speed(axis, targetvel)

            self.pmac.move_axis(axis, targetpos)
            self.ui.le_reldisp.setText('')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to move to target position.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def releaseAccessToMovement(self):
        """Check homing status and enable movement."""
        if not self.pmac.connected:
            return

        try:
            list_of_axis = self.pmac.commands.list_of_axis

            if not self.pmac.connected:
                for axis in list_of_axis:
                    axis_led = getattr(self.ui, 'la_ledax' + str(axis))
                    axis_led.setEnabled(False)
                return

            item = 0
            homing_status = []
            for axis in list_of_axis:
                axis_led = getattr(self.ui, 'la_ledax' + str(axis))
                if self.pmac.axis_homing_status(axis):
                    self.ui.cmb_selectaxis.model().item(
                        item+1).setEnabled(True)
                    self.ui.cmb_selecttrigaxis.model().item(
                        item+1).setEnabled(True)
                    axis_led.setEnabled(True)
                    homing_status.append(True)
                else:
                    self.ui.cmb_selectaxis.model().item(
                        item+1).setEnabled(False)
                    self.ui.cmb_selecttrigaxis.model().item(
                        item+1).setEnabled(False)
                    axis_led.setEnabled(False)
                    homing_status.append(False)
                item += 1

            if any(homing_status):
                self.setMovementEnabled(True)
                self.setTriggerEnabled(True)
                self.updateVelocityAndPosition()
                self.updateTrigAxisVelocity()
            else:
                self.setMovementEnabled(False)
                self.setTriggerEnabled(False)

            if all(homing_status):
                self.homing = True
            else:
                self.homing = False

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def resetAxisLimits(self):
        """Reset axis limits."""
        try:
            neg_list = self.pmac.commands.i_softlimit_neg_list
            pos_list = self.pmac.commands.i_softlimit_pos_list

            if self.pmac.get_response(self.pmac.set_par(neg_list[0], 0)):
                self.ui.le_minax1.setText('')

            if self.pmac.get_response(self.pmac.set_par(pos_list[0], 0)):
                self.ui.le_maxax1.setText('')

            if self.pmac.get_response(self.pmac.set_par(neg_list[1], 0)):
                self.ui.le_minax2.setText('')

            if self.pmac.get_response(self.pmac.set_par(pos_list[1], 0)):
                self.ui.le_maxax2.setText('')

            if self.pmac.get_response(self.pmac.set_par(neg_list[2], 0)):
                self.ui.le_minax3.setText('')

            if self.pmac.get_response(self.pmac.set_par(pos_list[2], 0)):
                self.ui.le_maxax3.setText('')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to reset axis limitis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def selectedAxis(self):
        """Return the selected axis."""
        axis_str = self.ui.cmb_selectaxis.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in self.pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def selectedTriggerAxis(self):
        """Return the selected trigger axis."""
        axis_str = self.ui.cmb_selecttrigaxis.currentText()
        if axis_str == '':
            return None

        axis = int(axis_str[1])
        if axis in self.pmac.commands.list_of_axis:
            return axis
        else:
            return None

    def setAxisLimits(self):
        """Set axis limits."""
        try:
            neg_list = self.pmac.commands.i_softlimit_neg_list
            pos_list = self.pmac.commands.i_softlimit_pos_list
            cts_mm_axis = self.pmac.commands.CTS_MM_AXIS

            minax1 = _utils.getValueFromStringExpresssion(
                self.ui.le_minax1.text())
            maxax1 = _utils.getValueFromStringExpresssion(
                self.ui.le_maxax1.text())

            minax2 = _utils.getValueFromStringExpresssion(
                self.ui.le_minax2.text())
            maxax2 = _utils.getValueFromStringExpresssion(
                self.ui.le_maxax2.text())

            minax3 = _utils.getValueFromStringExpresssion(
                self.ui.le_minax3.text())
            maxax3 = _utils.getValueFromStringExpresssion(
                self.ui.le_maxax3.text())

            if minax1 is not None and maxax1 is not None:
                minax1 = minax1*cts_mm_axis[0]
                maxax1 = maxax1*cts_mm_axis[0]
                self.pmac.get_response(self.pmac.set_par(neg_list[0], minax1))
                self.pmac.get_response(self.pmac.set_par(pos_list[0], maxax1))

            if minax2 is not None and maxax2 is not None:
                minax2 = minax2*cts_mm_axis[1]
                maxax2 = maxax2*cts_mm_axis[1]
                self.pmac.get_response(self.pmac.set_par(neg_list[1], minax2))
                self.pmac.get_response(self.pmac.set_par(pos_list[1], maxax2))

            if minax3 is not None and maxax3 is not None:
                minax3 = minax3*cts_mm_axis[2]
                maxax3 = maxax3*cts_mm_axis[2]
                self.pmac.get_response(self.pmac.set_par(neg_list[2], minax3))
                self.pmac.get_response(self.pmac.set_par(pos_list[2], maxax3))

            self.ui.pbt_setlimits.setEnabled(False)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Could not set axis limits.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def setAxisLimitsEnabled(self, enabled):
        """Enable/Disable axis limits controls."""
        self.ui.gb_limits.setEnabled(enabled)
        self.ui.pbt_setlimits.setEnabled(enabled)
        self.ui.pbt_resetlimits.setEnabled(enabled)

    def setHomingEnabled(self, enabled):
        """Enable/Disable homing controls."""
        self.ui.gb_homing.setEnabled(enabled)
        self.ui.pbt_homing.setEnabled(enabled)

    def setMovementEnabled(self, enabled):
        """Enable/Disable movement controls."""
        self.ui.gb_moveaxis.setEnabled(enabled)
        self.ui.pbt_move.setEnabled(enabled)
        self.ui.pbt_stop.setEnabled(enabled)

    def setTriggerEnabled(self, enabled):
        """Enable/Disable trigger controls."""
        self.ui.gb_trigger.setEnabled(enabled)
        self.ui.pbt_trigandmove.setEnabled(enabled)
        self.ui.pbt_trigstop.setEnabled(enabled)

    def setVelocityPositionStrFormat(self, line_edit):
        """Set the velocity and position string format."""
        try:
            if not _utils.setFloatLineEditText(line_edit, precision=3):
                self.updateVelocityAndPosition()
        except Exception:
            pass

    def setTriggerandMove(self, axis):
        """Set trigger and move axis."""
        self.stop_trigger = False
        axis = self.selectedTriggerAxis()
        if axis is None:
            return

        try:
            start = _utils.getValueFromStringExpresssion(
                self.ui.le_trigstart.text())
            if start is None:
                return

            step = _utils.getValueFromStringExpresssion(
                self.ui.le_trigstep.text())
            if step is None:
                return

            end = _utils.getValueFromStringExpresssion(
                self.ui.le_trigend.text())
            if end is None:
                return

            targetvel = _utils.getValueFromStringExpresssion(
                self.ui.le_trigvel.text())
            if targetvel is None:
                return

            npts = _np.abs(_np.ceil(round((end - start) / step, 4) + 1))

            velocity = self.pmac.get_velocity(axis)
            if targetvel != velocity:
                self.pmac.set_axis_speed(axis, targetvel)

            self.pmac.set_trigger(axis, start, step, 10, npts, 1)

            if self.stop_trigger:
                return

            self.pmac.move_axis(axis, start)
            while ((self.pmac.axis_status(axis) & 1) == 0 and
                   self.stop_trigger is False):
                _QApplication.processEvents()

            if self.stop_trigger:
                return

            if not self.ui.chb_trigpause.isChecked():
                self.pmac.move_axis(axis, end)
            else:
                pos_list = _np.linspace(start, end, npts)
                delay = self.ui.sbd_trigdelay.value()
                for pos in pos_list:
                    if self.stop_trigger:
                        return

                    self.pmac.move_axis(axis, pos)
                    while ((self.pmac.axis_status(axis) & 1) == 0 and
                           self.stop_trigger is False):
                        _QApplication.processEvents()
                    for i in range(100):
                        _QApplication.processEvents()
                        _time.sleep(delay/100)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure trigger and move axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def startHoming(self):
        """Homing of the selected axes."""
        try:
            axis_homing_mask = 0
            list_of_axis = self.pmac.commands.list_of_axis

            for axis in list_of_axis:
                obj = getattr(self.ui, 'chb_homingax' + str(axis))
                val = int(obj.isChecked())
                axis_homing_mask += (val << (axis-1))

            self.pmac.align_bench(axis_homing_mask)
            _time.sleep(self._align_bench_time_interval)

            while int(self.pmac.read_response(
                    self.pmac.commands.prog_running)) == 1:
                _time.sleep(self._align_bench_time_interval)
            else:
                self.releaseAccessToMovement()
                for axis in list_of_axis:
                    obj = getattr(self.ui, 'chb_homingax' + str(axis))
                    obj.setChecked(False)
                _QApplication.processEvents()
                msg = 'Finished homing of the selected axes.'
                _QMessageBox.information(self, 'Homing', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Homing failed.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def stopAllAxis(self):
        """Stop all axis."""
        try:
            self.pmac.stop_all_axis()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

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

    def stopTriggerAxis(self):
        """Stop the selected trigger axis."""
        self.stop_trigger = True
        try:
            axis = self.selectedTriggerAxis()
            if axis is None:
                return
            self.pmac.stop_axis(axis)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop axis.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def updateRelDisp(self):
        """Update relative displacement value."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return

            position = self.pmac.get_position(axis)
            targetpos = _utils.getValueFromStringExpresssion(
                self.ui.le_targetpos.text())
            reldisp = targetpos - position
            self.ui.le_reldisp.setText(self._position_format.format(reldisp))

        except Exception:
            pass

    def updateTargetPos(self):
        """Update target position value."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return

            position = self.pmac.get_position(axis)
            reldisp = _utils.getValueFromStringExpresssion(
                self.ui.le_reldisp.text())
            targetpos = position + reldisp
            self.ui.le_targetpos.setText(self._position_format.format(
                targetpos))

        except Exception:
            pass

    def updateTrigAxisVelocity(self):
        """Update velocity for the trigger axis."""
        try:
            axis = self.selectedTriggerAxis()
            if axis is None:
                return

            velocity = self.pmac.get_velocity(axis)
            self.ui.le_trigvel.setText(self._position_format.format(velocity))
        except Exception:
            pass

    def updateVelocityAndPosition(self):
        """Update velocity and position values for the selected axis."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return

            velocity = self.pmac.get_velocity(axis)
            position = self.pmac.get_position(axis)
            self.ui.le_targetvel.setText(self._position_format.format(
                velocity))
            self.ui.le_reldisp.setText(self._position_format.format(0))
            self.ui.le_targetpos.setText(self._position_format.format(
                position))

            self.ui.la_targetvelunit.setText(self._axis_unit[axis] + '/s')
            self.ui.la_reldispunit.setText(self._axis_unit[axis])
            self.ui.la_targetposunit.setText(self._axis_unit[axis])
        except Exception:
            pass
