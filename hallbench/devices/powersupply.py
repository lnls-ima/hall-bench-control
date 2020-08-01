# -*- coding: utf-8 -*-

import time as _time
from imautils.devices import pydrs_firmware_updated as _pydrs


class PowerSupply(_pydrs.SerialDRS):
    """Power Supply class."""

    def __init__(
            self, port, ps_address, kp, ki, slope,
            dclink=False, dclink_address=None, dclink_voltage=None,
            bipolar=False, current_min=0, current_max=0,
            dsp_class=3, dsp_id=0):

        self.port = port
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

        super().__init__()

    @property
    def connected(self):
        """Return True if the device is connected, False otherwise."""
        if self.ser.is_open:
            return True

        return False

    def connect(self):
        self.Connect(self.port)

    def set_address_ps(self):
        if not self.connected:
            print('Serial port is closed.')
            return False

        self.SetSlaveAdd(self.ps_address)
        return True

    def set_address_dclink(self):
        if not self.connected:
            print('Serial port is closed.')
            return False

        self.SetSlaveAdd(self.dclink_address)
        return True

    def set_op_mode(self, mode):
        """Sets power supply operation mode.

        Args:
            mode (str): SlowRef or Cycle.

        Returns:
            True in case of success.
            False otherwise."""

        if mode not in ['SlowRef', 'Cycle']:
            raise Exception('Invalid power supply mode.')

        self.select_op_mode(mode)
        _time.sleep(0.1)

        if self.read_ps_opmode() == mode:
            return True

        print('Failed to set operation mode.')
        return False

    def check_interlocks(self):
        """Check power supply interlocks."""
        status_interlocks = self.read_ps_softinterlocks()
        if status_interlocks != 0:
            print('Software interlock active.')
            return False

        status_interlocks = self.read_ps_hardinterlocks()
        if status_interlocks != 0:
            print('Hardware interlock active.')
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

    def configure_ps_monopolar(self, force=False):
        if not self.bipolar:
            print('Not implemented for monopolar power supply.')
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
                if not self.turn_off_ps():
                    return False

                if not self.turn_off_dclink():
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

    def configure_ps_bipolar(self, force=False):
        if not self.bipolar:
            print('Not implemented for monopolar power supply.')
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
                if not self.turn_off_ps():
                    return False

                if not self.turn_off_dclink():
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

    def turn_on_dclink(self, maxiter=300, tol=1):
        if self.dclink is None:
            return True

        self.set_address_dclink()
        dclink_on = self.read_ps_onoff()

        # Turn on DC link
        if not dclink_on:
            self.turn_on()
            _time.sleep(1.2)
            if not self.read_ps_onoff():
                print('Failed to turn on capacitor bank.')
                self.turn_off_dclink()
                return False

        # Closing DC link Loop
        self.closed_loop()
        _time.sleep(1)
        if self.read_ps_openloop():
            print('Failed to close capacitor bank loop.')
            self.turn_off_dclink()
            return False

        # Operation mode selection for SlowRef
        if not self.set_op_mode('SlowRef'):
            self.turn_off_dclink()
            return False

        self.set_slowref(self.dclink_voltage)
        _time.sleep(1)

        for _ in range(maxiter):
            voltage = self.read_vdclink()
            if abs(voltage - self.dclink_voltage) <= tol:
                return True
            _time.sleep(0.1)

        print('Failed to set voltage of capacitor bank.')
        self.turn_off_dclink()
        return False

    def turn_on_ps(self):
        self.set_address_ps()
        ps_on = self.read_ps_onoff()

        if not ps_on:
            self.turn_on()
            _time.sleep(1.2)
            if not self.read_ps_onoff():
                print('Failed to turn on power supply.')
                self.turn_off_ps()
                self.turn_off_dclink()
                return False

        # Closed Loop
        self.closed_loop()
        _time.sleep(1.2)
        if self.read_ps_openloop() == 1:
            print('Failed to close power supply loop.')
            self.turn_off_ps()
            self.turn_off_dclink()
            return False

        return True

    def turn_off_dclink(self):
        if self.dclink is None:
            return True

        self.set_address_dclink()
        self.turn_off()
        _time.sleep(1.2)

        if self.read_ps_onoff():
            print('Failed to turn off capacitor bank.')
            return False

        return True

    def turn_off_ps(self):
        self.set_address_ps()
        self.turn_off()
        _time.sleep(1.2)

        if self.read_ps_onoff():
            print('Failed to turn off power supply.')
            return False

        return True

    def start_power_supply(self):
        """Starts the Power Supply."""
        if not self.set_address_ps():
            return False

        if not self.check_interlocks():
            return False

        if self.dclink:
            if not self.turn_on_dclink():
                return False

        if self.turn_on_ps():
            return True

        return False

    def stop_power_supply(self):
        """Stops the Power Supply."""
        if self.turn_off_ps() and self.turn_off_dclink():
            return True

        return False

    def configure_pid(self):
        """Set power supply PID configurations."""
        if not self.set_address_ps():
            return False

        self.set_dsp_coeffs(
            self.dsp_class, self.dsp_id,
            [self.kp, self.ki, self.umax, self.umin])

        return True

    def set_current(self, setpoint, maxiter=300, tol=0.5):
        """Changes current setpoint.

        Args:
            setpoint (float): current setpoint [A]."""
        if not self.set_address_ps():
            return False

        if not self.read_ps_onoff():
            print('Power supply is off.')
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
            except Exception:
                pass
            _time.sleep(0.1)

        print('Failed to change current setpoint.')
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
            print('Current above maximum value.')
            return False

        if float(current) < self.current_min:
            print('Current below minimum value.')
            return False

        return True

    def cycling(
            self, sig_type, num_cycles, freq, amplitude,
            offset, aux0, aux1, aux2, aux3):
        """Initializes power supply cycling routine."""
        if not self.set_address_ps():
            return False

        if not self.verify_current_limits(amplitude):
            return False

        self.cfg_siggen(
            sig_type, num_cycles, freq, amplitude, offset,
            aux0, aux1, aux2, aux3)

        if sig_type == 2:
            duration = num_cycles*(aux0 + aux1 + aux2)
        else:
            duration = (num_cycles+1)/freq

        # set mode to Cycle
        if not self.set_op_mode('Cycle'):
            return False

        self.enable_siggen()

        deadline = _time.monotonic() + duration

        t = _time.monotonic()
        while t < deadline:
            _time.sleep(0.1)
            t = _time.monotonic()

        self.disable_siggen()

        # returns to mode SlowRef
        if not self.set_op_mode('SlowRef'):
            return False

        return True
