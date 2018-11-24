devtools::document()
devtools::check(manual = T)

# generate reference manual
R CMD Rd2pdf ./ --force -o manual.pdf

devtools::install()
library("pros")