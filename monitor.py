# -*- coding: utf-8 -*-

"""Run the Hall bench control application."""

from hallbench.gui import monitorapp as _app


THREAD = True


if THREAD:
    thread = _app.run_in_thread()
else:
    _app.run()
