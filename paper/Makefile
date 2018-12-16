NAME := project

default:
	pdflatex --file-line-error $(NAME).tex
	bibtex $(NAME)

clean:
	rm $(NAME).aux
	rm $(NAME).log
	rm $(NAME).out
	rm $(NAME).pdf