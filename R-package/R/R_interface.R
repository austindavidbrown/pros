#' Fit
#' 
#' The fit function for a specific lambda value
#' 
#' @param X the data matrix
#' @param y the response vector response
#' @param alpha convex combination
#' @param lambda The Lagrangian penalization value
#' @param algorithm the optimization algorithm
#' 
#' @return 
#' A class \code{pros} with
#'
#' @export
pros = function(X, y, alpha = c(1, 0, 0, 0, 0, 0, 0), lambda, algorithm = "proximal_gradient_cd", max_iter = 10000, tolerance = 10^(-7)) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector

  if (length(alpha) != 7) {
    stop("alpha needs to be length 7")
  }
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  B = .Call("R_fit", as.matrix(X), y, alpha, as.double(lambda), toString(algorithm))

  res = list("B" = B, "intercept" = mean(y))
  class(res) = "pros"
  return ( res )
}


#' Prediction
#' 
#' The Prediction function
#' 
#' @param prosObj an object of class \code{pros}
#' @param X the data matrix 
#' 
#' @return 
#' A \code{vector} of predictions
#'
#' @export
predict.pros = function(prosObj, X) {
  B = matrix(as.vector(t(prosObj$B)), ncol = 1) # convert to column vector
  intercept = prosObj$intercept

  return ( .Call("R_predict", B, as.double(intercept), as.matrix(X)) )
}


#' Cross Validation
#' 
#' The K-fold cross-validation function
#' 
#' @param X the data matrix
#' @param y the response vector
#' @param K_fold partition size 
#' @param alpha convex combination
#' @param lambdas A vector of Lagrangian penalization values to be evaluated
#' @param algorithm the optimization algorithm
#' 
#' @return 
#' A class \code{cv_pros} with
#' \itemize{
#'   \item \code{best_lambda} the  best lambda.
#'   \item \code{lambdas} the lambda values
#'   \item \code{risks} the cross-validation risks
#' }
#'
#' @export
cv.pros = function(X, y, K_fold = 5, alpha = c(1, 0, 0, 0, 0, 0, 0), lambdas = seq(10^(-7), 1, .1), algorithm = "proximal_gradient_cd", max_iter = 10000, tolerance = 10^(-7)) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  res = .Call("R_cross_validation", as.matrix(X), y, as.double(K_fold), alpha, as.vector(lambdas), toString(algorithm))
  res$X = X
  res$y = y
  res$alpha = alpha
  class(res) = "cv_pros"
  return ( res )
}

#' CV Prediction
#' 
#' The Prediction function
#' 
#' @param cv_prosObj an object of class \code{cv_pros}
#' @param X_new the data matrix
#' 
#' @return 
#' A \code{vector} of predictions
#'
#' @export
predict.cv_pros = function(cv_prosObj, X_new) {
  X = cv_prosObj$X
  y = cv_prosObj$y
  alpha = cv_prosObj$alpha
  lambda = cv_prosObj$best_lambda
  fit = pros(X, y, alpha, lambda)
  res = predict(fit, X_new)
  return ( res )
}


###
# Test
###
test = function() {
  system("R CMD SHLIB R_interface.cpp")
  dyn.load("R_interface.so")

  X_train = read.csv("../../X_train.csv", header = F)
  y_train = read.csv("../../y_train.csv", header = F)

  X_test = read.csv("../../X_test.csv", header = F)
  y_test = read.csv("../../y_test.csv", header = F)

  ###
  # fit test
  ###
  .Call("R_fit", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1), matrix(as.vector(t(c(1, 0, 0, 0, 0, 0, 0))), ncol = 1), as.double(.01), toString("proximal_gradient_cd"))
  fit = pros(X_train, y_train, lambda = .1)
  fit

  ###
  # predict test
  ###
  .Call("R_predict", matrix(c(1, 1, 1, 1, 1), ncol = 1), as.double(mean(t(y_test))), as.matrix(X_test))
  pred = predict(fit, X_test)
  pred

  ###
  # CV test
  ###
  .Call("R_cross_validation", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1), as.double(5), 
        matrix(as.vector(t(c(1, 0, 0, 0, 0, 0, 0))), ncol = 1), c(.01, .5, 1), toString("subgradient_cd"))
  cv = cv.pros(X_train, y_train)
  print(cv)

  predict(cv, X_test)

}