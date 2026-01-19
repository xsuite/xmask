DOCS= corr_tripD1_procedure corr_MB_procedure

all: $(DOCS:%=%.pdf)

%.pdf: %.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $<

clean:
	latexmk -C

.PHONY: all clean
