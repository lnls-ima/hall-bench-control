# -*- coding: utf-8 -*-

"""Motors widget for the Hall Bench Control application."""

import time as _time
import numpy as _np
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.currentpositionwidget import CurrentPositionWidget \
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
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.current_position_widget = _CurrentPositionWidget(self)
        self.ui.position_lt.addWidget(self.current_position_widget)

        # variables initialization
        self.homing = False
        self.stop_trigger = False

        # disable combo box itens
        for item in range(self.ui.selectaxis_cmb.count()):
            self.ui.selectaxis_cmb.model().item(item).setEnabled(False)
        for item in range(self.ui.selecttrigaxis_cmb.count()):
            self.ui.selecttrigaxis_cmb.model().item(item).setEnabled(False)

        # disable trigger delay
        self.ui.trigpause_chb.setChecked(False)
        self.ui.trigdelay_sb.setEnabled(False)

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
                self.releaseAccessToMovement()

            else:
                self.setHomingEnabled(False)
                self.setAxisLimitsEnabled(False)
                self.setMovementEnabled(False)
                self.setTriggerEnabled(False)
                message = 'Failed to activate bench.'
                _QMessageBox.critical(
                    self, 'Failure', message, _QMessageBox.Ok)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            event.accept()
        except Exception:
            event.accept()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
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

        self.ui.trigvel_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.trigvel_le))
        self.ui.trigstart_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.trigstart_le))
        self.ui.trigstep_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.trigstep_le))
        self.ui.trigend_le.editingFinished.connect(
            lambda: self.setVelocityPositionStrFormat(self.ui.trigend_le))

        self.ui.trigstart_le.editingFinished.connect(self.fixPositionValues)
        self.ui.trigstep_le.editingFinished.connect(self.fixPositionValues)
        self.ui.trigend_le.editingFinished.connect(self.fixPositionValues)

        self.ui.reldisp_le.editingFinished.connect(self.updateTargetPos)
        self.ui.targetpos_le.editingFinished.connect(self.updateRelDisp)

        self.ui.selectaxis_cmb.currentIndexChanged.connect(
            self.updateVelocityAndPosition)

        self.ui.selecttrigaxis_cmb.currentIndexChanged.connect(
            self.updateTrigAxisVelocity)
        self.ui.trigpause_chb.stateChanged.connect(self.enableTriggerDelay)

        self.ui.activate_btn.clicked.connect(self.activateBench)
        self.ui.stopall_btn.clicked.connect(self.stopAllAxis)
        self.ui.killall_btn.clicked.connect(self.killAllAxis)
        self.ui.homing_btn.clicked.connect(self.startHoming)
        self.ui.setlimits_btn.clicked.connect(self.setAxisLimits)
        self.ui.resetlimits_btn.clicked.connect(self.resetAxisLimits)
        self.ui.move_btn.clicked.connect(self.moveToTarget)
        self.ui.stop_btn.clicked.connect(self.stopAxis)
        self.ui.trigandmove_btn.clicked.connect(self.setTriggerandMove)
        self.ui.trigstop_btn.clicked.connect(self.stopTriggerAxis)

    def enableTriggerDelay(self):
        """Enable or disable trigger delay."""
        if self.ui.trigpause_chb.isChecked():
            self.ui.trigdelay_sb.setEnabled(True)
        else:
            self.ui.trigdelay_sb.setEnabled(False)

    def fixPositionValues(self):
        """Fix step and end position value."""
        start_text = self.ui.trigstart_le.text()
        if not bool(start_text and start_text.strip()):
            return
        start = float(start_text)

        step_text = self.ui.trigstep_le.text()
        if not bool(step_text and step_text.strip()):
            return
        step = float(step_text)

        end_text = self.ui.trigend_le.text()
        if not bool(end_text and end_text.strip()):
            return
        end = float(end_text)

        if start is not None and step is not None and end is not None:
            npts = _np.abs(_np.round(round((end - start) / step, 4) + 1))
            if start <= end and step < 0:
                self.ui.trigstep_le.setText('')
                return
            elif start > end and step > 0:
                self.ui.trigstep_le.setText('')
                return
            elif start <= end:
                corrected_step = _np.abs(step)
                corrected_end = start + (npts-1)*corrected_step
            else:
                corrected_step = _np.abs(step)*(-1)
                corrected_end = start + (npts-1)*corrected_step

            self.ui.trigstep_le.setText('{0:0.4f}'.format(corrected_step))
            self.ui.trigend_le.setText('{0:0.4f}'.format(corrected_end))

    def killAllAxis(self):
        """Kill all axis."""
        try:
            self.pmac.kill_all_axis()
            self.setHomingEnabled(False)
            self.setAxisLimitsEnabled(False)
            self.setMovementEnabled(False)
            self.setTriggerEnabled(False)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def moveToTarget(self, axis):
        """Move Hall probe to target position."""
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
            self.ui.reldisp_le.setText('')

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def releaseAccessToMovement(self):
        """Check homing status and enable movement."""
        if not self.pmac.connected:
            return

        try:
            list_of_axis = self.pmac.commands.list_of_axis

            if not self.pmac.connected:
                for axis in list_of_axis:
                    axis_led = getattr(self.ui, 'ledax' + str(axis) + '_la')
                    axis_led.setEnabled(False)
                return

            item = 0
            homing_status = []
            for axis in list_of_axis:
                axis_led = getattr(self.ui, 'ledax' + str(axis) + '_la')
                if self.pmac.axis_homing_status(axis):
                    self.ui.selectaxis_cmb.model().item(
                        item+1).setEnabled(True)
                    self.ui.selecttrigaxis_cmb.model().item(
                        item+1).setEnabled(True)
                    axis_led.setEnabled(True)
                    homing_status.append(True)
                else:
                    self.ui.selectaxis_cmb.model().item(
                        item+1).setEnabled(False)
                    self.ui.selectrigtaxis_cmb.model().item(
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

            self.window().updateMainTabStatus()

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def resetAxisLimits(self):
        """Reset axis limits."""
        try:
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

    def selectedTriggerAxis(self):
        """Return the selected trigger axis."""
        axis_str = self.ui.selecttrigaxis_cmb.currentText()
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

            if (len(self.ui.minax1_le.text()) != 0 and
               len(self.ui.maxax1_le.text()) != 0):
                minax1 = float(self.ui.minax1_le.text())*cts_mm_axis[0]
                maxax1 = float(self.ui.maxax1_le.text())*cts_mm_axis[0]
                self.pmac.get_response(self.pmac.set_par(neg_list[0], minax1))
                self.pmac.get_response(self.pmac.set_par(pos_list[0], maxax1))

            if (len(self.ui.minax2_le.text()) != 0 and
               len(self.ui.maxax2_le.text()) != 0):
                minax2 = float(self.ui.minax2_le.text())*cts_mm_axis[1]
                maxax2 = float(self.ui.maxax2_le.text())*cts_mm_axis[1]
                self.pmac.get_response(self.pmac.set_par(neg_list[1], minax2))
                self.pmac.get_response(self.pmac.set_par(pos_list[1], maxax2))

            if (len(self.ui.minax3_le.text()) != 0 and
               len(self.ui.maxax3_le.text()) != 0):
                minax3 = float(self.ui.minax3_le.text())*cts_mm_axis[2]
                maxax3 = float(self.ui.maxax3_le.text())*cts_mm_axis[2]
                self.pmac.get_response(self.pmac.set_par(neg_list[2], minax3))
                self.pmac.get_response(self.pmac.set_par(pos_list[2], maxax3))

        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Could not set axis limits.', _QMessageBox.Ok)

    def setAxisLimitsEnabled(self, enabled):
        """Enable/Disable axis limits controls."""
        self.ui.limits_gb.setEnabled(enabled)
        self.ui.setlimits_btn.setEnabled(enabled)
        self.ui.resetlimits_btn.setEnabled(enabled)

    def setAxisLimitsStrFormat(self, obj):
        """Set the axis limit string format."""
        try:
            value = float(obj.text())
            obj.setText(self._position_format.format(value))
        except Exception:
            obj.setText('')

    def setHomingEnabled(self, enabled):
        """Enable/Disable homing controls."""
        self.ui.homing_gb.setEnabled(enabled)
        self.ui.homing_btn.setEnabled(enabled)

    def setMovementEnabled(self, enabled):
        """Enable/Disable movement controls."""
        self.ui.moveaxis_gb.setEnabled(enabled)
        self.ui.move_btn.setEnabled(enabled)
        self.ui.stop_btn.setEnabled(enabled)

    def setTriggerEnabled(self, enabled):
        """Enable/Disable trigger controls."""
        self.ui.trigger_gb.setEnabled(enabled)
        self.ui.trigandmove_btn.setEnabled(enabled)
        self.ui.trigstop_btn.setEnabled(enabled)

    def setVelocityPositionStrFormat(self, obj):
        """Set the velocity and position string format."""
        try:
            value = float(obj.text())
            obj.setText(self._position_format.format(value))
        except Exception:
            self.updateVelocityAndPosition()

    def setTriggerandMove(self, axis):
        """Set trigger and move axis."""
        self.stop_trigger = False
        axis = self.selectedTriggerAxis()
        if axis is None:
            return

        try:
            start_text = self.ui.trigstart_le.text()
            if len(start_text) == 0:
                return
            start = float(start_text)

            step_text = self.ui.trigstep_le.text()
            if len(step_text) == 0:
                return
            step = float(step_text)

            end_text = self.ui.trigend_le.text()
            if len(end_text) == 0:
                return
            end = float(end_text)

            targetvel_text = self.ui.trigvel_le.text()
            if len(targetvel_text) == 0:
                return
            targetvel = float(targetvel_text)

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

            if not self.ui.trigpause_chb.isChecked():
                self.pmac.move_axis(axis, end)
            else:
                pos_list = _np.linspace(start, end, npts)
                delay = self.ui.trigdelay_sb.value()
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

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def startHoming(self):
        """Homing of the selected axes."""
        try:
            axis_homing_mask = 0
            list_of_axis = self.pmac.commands.list_of_axis

            for axis in list_of_axis:
                obj = getattr(self.ui, 'homingax' + str(axis) + '_chb')
                val = int(obj.isChecked())
                axis_homing_mask += (val << (axis-1))

            self.pmac.align_bench(axis_homing_mask)
            _time.sleep(self._align_bench_time_interval)

            while int(self.pmac.read_response(
                    self.pmac.commands.prog_running)) == 1:
                _time.sleep(self._align_bench_time_interval)
            else:
                self.releaseAccessToMovement()
                message = 'Finished homing of the selected axes.'
                _QMessageBox.information(
                    self, 'Homing', message, _QMessageBox.Ok)
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Homing failed.', _QMessageBox.Ok)

    def stopAllAxis(self):
        """Stop all axis."""
        try:
            self.pmac.stop_all_axis()

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

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

    def stopTriggerAxis(self):
        """Stop the selected trigger axis."""
        self.stop_trigger = True
        try:
            axis = self.selectedTriggerAxis()
            if axis is None:
                return
            self.pmac.stop_axis(axis)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def updateRelDisp(self):
        """Update relative displacement value."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return

            position = self.pmac.get_position(axis)
            targetpos = float(self.ui.targetpos_le.text())
            reldisp = targetpos - position
            self.ui.reldisp_le.setText(self._position_format.format(reldisp))

        except Exception:
            pass

    def updateTargetPos(self):
        """Update target position value."""
        try:
            axis = self.selectedAxis()
            if axis is None:
                return

            position = self.pmac.get_position(axis)
            reldisp = float(self.ui.reldisp_le.text())
            targetpos = position + reldisp
            self.ui.targetpos_le.setText(self._position_format.format(
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
            self.ui.trigvel_le.setText(self._position_format.format(velocity))
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
            self.ui.targetvel_le.setText(self._position_format.format(
                velocity))
            self.ui.reldisp_le.setText(self._position_format.format(0))
            self.ui.targetpos_le.setText(self._position_format.format(
                position))

            self.ui.targetvelunit_la.setText(self._axis_unit[axis] + '/s')
            self.ui.reldispunit_la.setText(self._axis_unit[axis])
            self.ui.targetposunit_la.setText(self._axis_unit[axis])
        except Exception:
            pass
