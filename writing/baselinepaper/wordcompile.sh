#!/bin/bash
pandoc \
	--reference-doc=template.docx \
	--filter pandoc-crossref \
	--citeproc \
	--bibliography=library.bib \
	-f latex \
	-t docx \
	--verbose \
	--number-sections \
	--csl=nature.csl \
	main.tex \
	-o main.docx
