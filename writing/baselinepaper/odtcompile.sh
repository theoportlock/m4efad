#!/bin/bash
pandoc \
	--reference-doc=template.odt \
	--filter pandoc-crossref \
	--citeproc \
	--bibliography=library.bib \
	-f latex \
	-t odt \
	--verbose \
	--number-sections \
	--csl=nature.csl \
	main.tex \
	-o main.odt
