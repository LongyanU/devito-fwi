
SRCDIR = src
TARGETDIR = bin

all: BFM2D

BFM2D:
	cd $(SRCDIR); make bfm2d
	mkdir -p $(TARGETDIR)
	cd $(SRCDIR); mv bfm2d ../$(TARGETDIR)
