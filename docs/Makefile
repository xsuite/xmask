DOC= corr_tripD1_procedure

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $<

clean:
	latexmk -C

.PHONY: all clean
