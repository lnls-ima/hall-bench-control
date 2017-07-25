# -*- coding: utf-8 -*-
"""
Variables and constants to be used in Agilent 33521A
"""
class ListOfCommands(object):
    def __init__(self):
        """ Initiate all variables """
        self._reset()
        
    def _reset(self):
        """ Reset Function"""
        self.reset = '*RST'
