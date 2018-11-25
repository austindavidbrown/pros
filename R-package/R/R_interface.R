#' Fit
#' 
#' The fit function for a specific lambda value
#' 
#' @param X the data matrix
#' @param y the vector response
#' @param lambda A lambda value
#' 
#' @return 
#' A class \code{pros} with
#'
#' @export
pros = function(X, y, lambda) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector
  B = .Call("R_sgcd", as.matrix(X), y, as.double(lambda))

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
#' @param y the vector response
#' @param K_fold partition size 
#' @param lambdas lambda values to be evaluated
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
cv.pros = function(X, y, K_fold, lambdas) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector

  res = .Call("R_cross_validation", as.matrix(X), y, as.double(K_fold), as.vector(lambdas))
  res$X = X
  res$y = y
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
  lambda = cv_prosObj$best_lambda
  fit = pros(X, y, lambda)
  res = predict(fit, X_new)
  return ( res )
}






###
# Test
###
test_ = function() {
  #system("R CMD SHLIB R_interface.cpp")
  #dyn.load("R_interface.so")

  X_train = read.csv("../../X_train.csv", header = F)
  y_train = read.csv("../../y_train.csv", header = F)

  X_test = read.csv("../../X_test.csv", header = F)
  y_test = read.csv("../../y_test.csv", header = F)

  ###
  # fit test
  ###
  lambda = .01
  .Call("R_sgcd", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1), as.double(lambda))
  fit = pros(X_train, y_train, .01)
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
  K_fold = 5
  lambdas = c(.01, .5, 1)

  .Call("R_cross_validation", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1), as.double(K_fold), as.vector(lambdas))
  cv = cv.pros(X_train, y_train, K_fold, lambdas)
  print(cv)

  predict(cv, X_test)

}