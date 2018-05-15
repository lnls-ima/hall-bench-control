# -*- coding: utf-8 -*-

"""Position widget for the Hall Bench Control application."""

from PyQt4 import QtGui as _QtGui
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class PositionWidget(_QtGui.QWidget):
    """Position Widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Setup the ui."""
        super(PositionWidget, self).__init__(parent)

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
            _QtGui.QApplication.processEvents()
        except Exception:
            pass
