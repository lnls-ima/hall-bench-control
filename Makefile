# Makefile for Hall Bench GUI

all:
	pyuic4 interface.ui -o interface.py
	pyrcc4 -py3 resource_file.qrc -o resource_file_rc.py

clean:
	-rm -rf interface.py resource_file_rc.py
