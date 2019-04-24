"""UDC communication.

Created on 9 de out de 2018
@author: Vitor Soares
"""

from . import interfaces as _interfaces


def UDC_factory(baseclass):
    """Create UDC class."""
    class UDC(baseclass):
        """Honeywell UDC control class."""

        def __init__(self, logfile=None):
            """Honeywell UDC control class.

            Args:
                logfile (str): log file path.
            """
            self.output1_register_address = 70
            self.output2_register_address = 382
            self.pv1_register_address = 72
            self.pv2_register_address = 74
            super().__init__(logfile)

        def read_output1(self):
            """Return controller output 1."""
            if self.output1_register_address is None:
                return None
            else:
                return self.read_from_device(self.output1_register_address)

        def read_output2(self):
            """Return controller output 2."""
            if self.output2_register_address is None:
                return None
            else:
                return self.read_from_device(self.output2_register_address)

        def read_pv1(self):
            """Return process variable."""
            if self.pv1_register_address is None:
                return None
            else:
                return self.read_from_device(self.pv1_register_address)

        def read_pv2(self):
            """Return process variable 2."""
            if self.pv2_register_address is None:
                return None
            else:
                return self.read_from_device(self.pv2_register_address)

    return UDC


UDCModBus = UDC_factory(_interfaces.ModBusInterface)
