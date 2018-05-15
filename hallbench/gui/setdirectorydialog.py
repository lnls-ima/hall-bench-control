# -*- coding: utf-8 -*-

"""Directory dialog for the Hall Bench Control application."""

from PyQt4 import QtGui as _QtGui
from PyQt4 import QtCore as _QtCore
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class SetDirectoryDialog(_QtGui.QDialog):
    """Directory dialog class for the Hall Bench Control application."""

    directoryChanged = _QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        """Setup the ui and create connections."""
        super(SetDirectoryDialog, self).__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.directory = None

        self.ui.changedir_btn.clicked.connect(self.changeLineEditDirectory)
        self.ui.dir_btnbox.accepted.connect(self.setDirectory)
        self.ui.dir_btnbox.rejected.connect(self.close)

    def changeLineEditDirectory(self):
        """Change the directory line edit text."""
        directory = _QtGui.QFileDialog.getExistingDirectory(
            self, caption='Set Directory', directory=self.directory,
            options=_QtGui.QFileDialog.ShowDirsOnly)

        if len(directory) != 0:
            self.ui.dir_le.setText(directory)

    def setDirectory(self):
        """Set directory."""
        self.directory = self.ui.dir_le.text()
        self.directoryChanged.emit(self.directory)
        self.close()

    def show(self, directory):
        """Update line edit text and show dialog."""
        self.ui.dir_le.setText(directory)
        super(SetDirectoryDialog, self).show()
