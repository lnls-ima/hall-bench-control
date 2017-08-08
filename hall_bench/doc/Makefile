# Makefile for Doxygen documentation

DOXYFILE = Doxyfile
PROJECT_DIR = ./../hall_bench
PROJECT_NUMBER = $(shell cat $(PROJECT_DIR)/VERSION)

html:
	( cat $(DOXYFILE) ; echo "PROJECT_NUMBER=$(PROJECT_NUMBER)" ) | doxygen -
	ln -s html/index.html documentation.html

clean:
	rm -rf html
	rm documentation.html
