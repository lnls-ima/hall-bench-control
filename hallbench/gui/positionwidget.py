# -*- coding: utf-8 -*-

"""Position widget for the Hall Bench Control application."""

from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QApplication as _QApplication,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class PositionWidget(_QWidget):
    """Position Widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

    @property
    def pmac(self):
        """Pmac object."""
        return self.window().devices.pmac

    def updatePositions(self):
        """Update axes positions."""
        if self.pmac is None:
            return

        try:
            for axis in self.pmac.commandslist_of_axis:
                pos = self.pmac.get_position(axis)
                le = getattr(self.ui, 'posax' + str(axis) + '_le')
                le.setText('{0:0.4f}'.format(pos))
            _QApplication.processEvents()
        except Exception:
            pass
