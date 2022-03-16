.PHONY: main.pdf all clean

all: ms.pdf

ms.pdf: ms.tex intro.tex methods.tex results.tex conclusions.tex  biblio.bib 
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make ms.tex

clean:
	latexmk -CA
