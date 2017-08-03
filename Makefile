# Makefile for Hall Bench GUI

all:
	pyuic4 interface.ui -o interface.py
	pyrcc4 -py3 resources.qrc -o resources_rc.py

clean:
	-rm -rf interface.py resources_rc.py
