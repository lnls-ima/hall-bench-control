# -*- coding: utf-8 -*-

import time as _time
import logging as _logging
from imautils.devices import pydrs_firmware_updated as _pydrs


class PowerSupply(_pydrs.SerialDRS):
    """Power Supply class."""

    def __init__(self, log=False):
        self.port = None
        self.ps_address = None
        self.kp = None
        self.ki = None
        self.slope = None
        self.dclink = None
        self.dclink_address = None
        self.dclink_voltage = None
        self.bipolar = None
        self.current_min = None
        self.current_max = None
        self.dsp_class = None
        self.dsp_id = None
        self.umax = None
        self.umin = None
        self._configured = False
        
        self.log = log
        self.logger = None
        self.log_events()
        
        super().__init__()

    @property
    def configured(self):
        """Return True if the device is configured, False otherwise."""
        return self._configured

    @property
    def connected(self):
        """Return True if the device is connected, False otherwise."""
        if self.ser.is_open:
            return True

        return False

    def log_events(self):
        """Prepare log file to save info, warning and error status."""
        if self.log:
            self.logger = _logging.getLogger()
            self.logger.setLevel(_logging.ERROR)

    def _turn_on_dclink(self, maxiter=300, tol=1, callback=None):
        if self.dclink is None:
            return True

        if not self.set_address_dclink():
            return False
        
        dclink_on = self.read_ps_onoff()

        # Turn on DC link
        if not dclink_on:
            self.turn_on()
            _time.sleep(1.2)
            if not self.read_ps_onoff():
                if self.logger is not None:
                    self.logger.error('Failed to turn on capacitor bank.')
                self._turn_off_dclink()
                return False

        # Closing DC link Loop
        self.closed_loop()
        _time.sleep(1)
        if self.read_ps_openloop():
            if self.logger is not None:
                self.logger.error('Failed to close capacitor bank loop.')
            self._turn_off_dclink()
            return False

        # Operation mode selection for SlowRef
        if not self.set_operation_mode('SlowRef'):
            self._turn_off_dclink()
            return False

        self.set_slowref(self.dclink_voltage)
        _time.sleep(1)

        for _ in range(maxiter):
            voltage = self.read_vdclink()
            if abs(voltage - self.dclink_voltage) <= tol:
                return True
            if callback is not None:
                callback()
            _time.sleep(0.1)

        if self.logger is not None:
            self.logger.error('Failed to set voltage of capacitor bank.')
        self._turn_off_dclink()
        return False

    def _turn_on_ps(self):       
        if not self.set_address_ps():
            return False
        
        ps_on = self.read_ps_onoff()

        if not ps_on:
            self.turn_on()
            _time.sleep(1.2)
            if not self.read_ps_onoff():
                if self.logger is not None:
                    self.logger.error('Failed to turn on power supply.')
                self._turn_off_ps()
                self._turn_off_dclink()
                return False

        # Closed Loop
        self.closed_loop()
        _time.sleep(1.2)
        if self.read_ps_openloop() == 1:
            if self.logger is not None:
                self.logger.error('Failed to close power supply loop.')
            self._turn_off_ps()
            self._turn_off_dclink()
            return False

        return True

    def _turn_off_dclink(self):     
        if self.dclink is None:
            return True

        self.set_address_dclink()
        self.turn_off()
        _time.sleep(1.2)

        if self.read_ps_onoff():
            if self.logger is not None:
                self.logger.error('Failed to turn off capacitor bank.')
            return False

        return True

    def _turn_off_ps(self):
        self.set_address_ps()
        self.turn_off()
        _time.sleep(1.2)

        if self.read_ps_onoff():
            if self.logger is not None:
                self.logger.error('Failed to turn off power supply.')
            return False

        return True

    def configure(
            self, ps_address, kp, ki, slope,
            dclink=False, dclink_address=None, dclink_voltage=None,
            bipolar=False, current_min=0, current_max=0,
            dsp_class=3, dsp_id=0):

        self.ps_address = ps_address
        self.kp = kp
        self.ki = ki
        self.slope = slope

        self.dclink = dclink
        self.dclink_address = dclink_address
        self.dclink_voltage = dclink_voltage
        self.bipolar = bipolar
        self.current_min = current_min
        self.current_max = current_max
        self.dsp_class = dsp_class
        self.dsp_id = dsp_id

        self.umax = 0.90
        if self.bipolar:
            self.umin = -0.90
        else:
            self.umin = 0
        
        self._configured = True
        
        return True

    def connect(self, port):       
        self.port = port
        return self.Connect(self.port)

    def set_address_ps(self):
        if not self.configured:
            if self.logger is not None:
                self.logger.error('Power supply not configured.')
            return False
        
        if not self.connected:
            if self.logger is not None:
                self.logger.error('Power supply serial port is closed.')
            return False

        self.SetSlaveAdd(self.ps_address)
        return True

    def set_address_dclink(self):       
        if not self.configured:
            if self.logger is not None:
                self.logger.error('Power supply not configured.')
            return False
        
        if not self.connected:
            if self.logger is not None:
                self.logger.error('Power supply serial port is closed.')
            return False

        self.SetSlaveAdd(self.dclink_address)
        return True

    def set_operation_mode(self, mode):
        """Sets power supply operation mode.

        Args:
            mode (str): SlowRef or Cycle.

        Returns:
            True in case of success.
            False otherwise."""
        if not self.connected:
            return False

        if mode not in ['SlowRef', 'Cycle']:
            if self.logger is not None:
                self.logger.error('Invalid power supply mode.')
            return False

        self.select_op_mode(mode)
        _time.sleep(0.1)

        if self.read_ps_opmode() == mode:
            return True

        if self.logger is not None:
            self.logger.error('Failed to set power supply operation mode.')
        return False

    def check_interlocks(self):
        """Check power supply interlocks."""   
        if not self.connected:
            return False
        
        status_interlocks = self.read_ps_softinterlocks()
        if status_interlocks != 0:
            if self.logger is not None:
                self.logger.error('Power supply software interlock active.')
            return False

        status_interlocks = self.read_ps_hardinterlocks()
        if status_interlocks != 0:
            if self.logger is not None:
                self.logger.error('Power supply hardware interlock active.')
            return False

        return True

    def reset_all_interlocks(self):
        """Resets power supply hardware/software interlocks"""           
        if self.dclink:
            self.set_address_dclink()
            self.reset_interlocks()

        self.set_address_ps()
        self.reset_interlocks()

        return True

    def configure_monopolar(self, force=False):
        if not self.bipolar:
            if self.logger is not None:
                self.logger.error(
                    'Not implemented for monopolar power supply.')
            return False

        if not self.set_address_ps():
            return False

        if not self.check_interlocks():
            return False

        ps_on = self.read_ps_onoff()

        if ps_on:
            if not force:
                return False

            else:
                if not self._turn_off_ps():
                    return False

                if not self._turn_off_dclink():
                    return False

        self.current_min = 0

        self.set_address_ps()
        self.set_param('Min_Ref', 0, 0)
        self.set_param('Min_Ref_OpenLoop', 0, 0)
        self.set_param('PWM_Min_Duty', 0, 0)
        self.set_param('PWM_Min_Duty_OpenLoop', 0, 0)
        self.set_dsp_coeffs(
            self.dsp_class, self.dsp_id,
            [self.kp, self.ki, self.umax, 0])

        _time.sleep(1)

        return True

    def configure_bipolar(self, force=False):
        if not self.bipolar:
            if self.logger is not None:
                self.logger.error(
                    'Not implemented for monopolar power supply.')
            return False

        if not self.set_address_ps():
            return False

        if not self.check_interlocks():
            return False

        ps_on = self.read_ps_onoff()

        if ps_on:
            if not force:
                return False

            else:
                if not self._turn_off_ps():
                    return False

                if not self._turn_off_dclink():
                    return False

        self.set_address_ps()
        self.set_param('Min_Ref', 0, self.current_min)
        self.set_param('Min_Ref_OpenLoop', 0, -40)
        self.set_param('PWM_Min_Duty', 0, self.umin)
        self.set_param('PWM_Min_Duty_OpenLoop', 0, -0.4)
        self.set_dsp_coeffs(
            self.dsp_class, self.dsp_id,
            [self.kp, self.ki, self.umax, self.umin])

        _time.sleep(1)

        return True

    def start_power_supply(self, callback=None):
        """Starts the Power Supply."""
        if not self.set_address_ps():
            return False

        if not self.check_interlocks():
            return False

        if self.dclink:
            if not self._turn_on_dclink(callback=callback):
                return False

        if self._turn_on_ps():
            return True

        return False

    def stop_power_supply(self):
        """Stops the Power Supply."""
        if self._turn_off_ps() and self._turn_off_dclink():
            return True

        return False

    def configure_pid(self):
        """Set power supply PID configurations."""
        if not self.set_address_ps():
            return False

        return self.set_dsp_coeffs(
            self.dsp_class, self.dsp_id,
            [self.kp, self.ki, self.umax, self.umin])

    def set_current(self, setpoint, maxiter=300, tol=0.5, callback=None):
        """Changes current setpoint.

        Args:
            setpoint (float): current setpoint [A]."""
        if not self.set_address_ps():
            return False

        if not self.read_ps_onoff():
            if self.logger is not None:
                self.logger.error('Power supply is off.')
            return False

        if not self.verify_current_limits(setpoint):
            return False

        self.set_slowref(setpoint)
        _time.sleep(0.1)

        for _ in range(maxiter):
            try:
                readback = round(float(self.read_iload1()), 3)
                if abs(readback - setpoint) <= tol:
                    return True
                if callback is not None:
                    callback()
            except Exception:
                pass
            _time.sleep(0.1)

        if self.logger is not None:
            self.logger.error('Failed to change power supply current.')
        return False

    def get_current(self):
        if not self.set_address_ps():
            return None

        return float(self.read_iload1())

    def verify_current_limits(self, current):
        """Check the limits of the current values set.

        Args:
            current (float): current value [A] to be verified.

        Return:
            True if current is within the limits, False otherwise.
        """
        if float(current) > self.current_max:
            if self.logger is not None:
                self.logger.error('Current above power supply maximum value.')
            return False

        if float(current) < self.current_min:
            if self.logger is not None:
                self.logger.error('Current below power supply minimum value.')
            return False

        return True

    def configure_siggen(
            self, sig_type, num_cycles, freq, amplitude,
            offset, aux0, aux1, aux2, aux3):
        """Configure signal generator."""
        if not self.set_address_ps():
            return False

        if not self.verify_current_limits(amplitude):
            return False

        return self.cfg_siggen(
            sig_type, num_cycles, freq, amplitude, offset,
            aux0, aux1, aux2, aux3)

    def cycle(
            self, sig_type, num_cycles, freq, amplitude,
            offset, aux0, aux1, aux2, aux3, callback=None):
        """Initializes power supply cycling routine."""
        if not self.configure_siggen(
                sig_type, num_cycles, freq, amplitude,
                offset, aux0, aux1, aux2, aux3):
            if self.logger is not None:
                self.logger.error('Failed to configured power supply siggen.')
            return False
        
        if sig_type == 2:
            duration = num_cycles*(aux0 + aux1 + aux2)
        else:
            duration = (num_cycles+1)/freq

        # set mode to Cycle
        if not self.set_operation_mode('Cycle'):
            return False

        self.enable_siggen()

        deadline = _time.monotonic() + duration

        t = _time.monotonic()
        while t < deadline:
            _time.sleep(0.1)
            if callback is not None:
                callback()
            t = _time.monotonic()

        self.disable_siggen()

        # returns to mode SlowRef
        if not self.set_operation_mode('SlowRef'):
            return False

        return True
