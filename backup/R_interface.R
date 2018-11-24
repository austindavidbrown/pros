
fit.pros = function(X, y, lambda) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector
  return ( .Call("R_fit", as.matrix(X), y, as.double(lambda)) )
}

predict.pros = function(X, B, intercept) {
  B = matrix(as.vector(t(B)), ncol = 1) # convert to column vector
  return ( .Call("R_predict", B, as.matrix(X), as.double(intercept)) )
}

cv.pros = function(X, y, K_fold, lambdas) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector
  return ( .Call("R_cross_validation", as.matrix(X), y, as.double(K_fold), as.vector(lambdas)) )
}


###
# Test
###
system("R CMD SHLIB R_interface.cpp")
dyn.load("R_interface.so")

X_train = read.csv("../../X_train.csv", header = F)
y_train = read.csv("../../y_train.csv", header = F)

X_test = read.csv("../../X_test.csv", header = F)
y_test = read.csv("../../y_test.csv", header = F)


lambda = .7
K_fold = 5
lambdas = c(.01, .5, 1)

intercept = mean(t(y_train))
B = c(1, 1, 2, 3, 5)
predict.pros(X_train, B, intercept)

fit.pros(X_train, y_train, .1)

cv.pros(X_train, y_train, K_fold, lambdas)