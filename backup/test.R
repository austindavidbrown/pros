

system("R CMD SHLIB R_interface.cpp")
dyn.load("R_interface.so")

X = matrix(c(5, 6, 7, 8), nrow = 2)
y = c(9, .5, .2, .1)
lambda = .7

.Call("R_sgcd", as.matrix(X), as.vector(y), as.double(lambda))