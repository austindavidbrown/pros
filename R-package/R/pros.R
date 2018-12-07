#' Pros
#' 
#' The fit function for a specific lambda value.
#' 
#' @param X the matrix of the data
#' @param y the vector of response values
#' @param alpha the convex combination of length 7 corresponding to the penalties:
#' \itemize{
#'   \item l1 penalty
#'   \item l2 penalty
#'   \item l4 penalty
#'   \item l6 penalty
#'   \item l8 penalty
#'   \item l10 penalty
#' }
#' @param lambda the dual penalization value
#' @param algorithm the optimization algorithm 
#' \itemize{
#'   \item proximal_gradient_cd
#'   \item subgradient_cd
#' }
#' @param max_iter maximum iterations. This also tunes the step size.
#' @param tolerance tolerance
#' @param random_seed random seed
#' 
#' @return 
#' A class \code{pros}
#'
#' @examples
#' fit = pros(X_train, y_train, lambda = .1)
#' pred = predict(fit, X_test)
#'
#' @export
pros = function(X, y, 
                alpha = c(1, 0, 0, 0, 0, 0), lambda, algorithm = "proximal_gradient_cd", 
                max_iter = 10000, tolerance = 10^(-3), random_seed = 0) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector

  if (length(alpha) != 6) {
    stop("alpha needs to be length 6")
  }
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  res = .Call("R_fit", as.matrix(X), y, 
            alpha, as.double(lambda), toString(algorithm), 
            as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  class(res) = "pros"
  return ( res )
}


#' Pros Prediction
#' 
#' The prediction function.
#' 
#' @param prosObj an object of class \code{pros}
#' @param X the matrix of the data to predict
#' 
#' @return 
#' A \code{vector} of prediction values.
#'
#' @examples
#' fit = pros(X_train, y_train, lambda = .1)
#' pred = predict(fit, X_test)
#'
#' @export
predict.pros = function(prosObj, X) {
  B = matrix(as.vector(t(prosObj$B)), ncol = 1) # convert to column vector
  intercept = prosObj$intercept
  return ( .Call("R_predict", B, as.double(intercept), as.matrix(X)) )
}


#' Cross-validation
#' 
#' The K-fold cross-validation function.
#' 
#' @param X the matrix of the data
#' @param y the vector of response values
#' @param alpha the convex combination of length 7 corresponding to the penalties:
#' \itemize{
#'   \item l1 penalty
#'   \item l2 penalty
#'   \item l4 penalty
#'   \item l6 penalty
#'   \item l8 penalty
#'   \item l10 penalty
#' }
#' @param lambdas A vector of dual penalization values to be evaluated
#' @param algorithm the optimization algorithm 
#' \itemize{
#'   \item proximal_gradient_cd
#'   \item subgradient_cd
#' }
#' @param max_iter maximum iterations. This also tunes the step size.
#' @param tolerance tolerance
#' @param random_seed random seed
#'
#' @return 
#' A class \code{cv_pros}
#'
#' @examples
#' cv = cv.pros(X_train, y_train)
#' pred = predict(cv, X_test)
#'
#' @export
cv.pros = function(X, y, 
                   K_fold = 10, alpha = c(1, 0, 0, 0, 0, 0), lambdas = seq(10^(-3), 1, .1), algorithm = "proximal_gradient_cd", 
                   max_iter = 10000, tolerance = 10^(-3), random_seed = 0) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  res = .Call("R_cross_validation", as.matrix(X), y, 
              as.double(K_fold), alpha, as.vector(lambdas), toString(algorithm), 
              as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  res$X = X
  res$y = y
  res$alpha = alpha
  res$max_iter = max_iter
  res$tolerance = tolerance
  res$random_seed = random_seed
  class(res) = "cv_pros"
  return ( res )
}

#' Cross-validation Prediction
#' 
#' The cross-validation prediction function.
#' 
#' @param cv_prosObj an object of class \code{cv_pros}
#' @param X_new the matrix of the data to predict
#' 
#' @return 
#' A \code{vector} of prediction values.
#'
#' @examples
#' cv = cv.pros(X_train, y_train)
#' pred = predict(cv, X_test)
#'
#' @export
predict.cv_pros = function(cv_prosObj, X_new) {
  X = cv_prosObj$X
  y = cv_prosObj$y
  alpha = cv_prosObj$alpha
  lambda = cv_prosObj$best_lambda
  max_iter = cv_prosObj$max_iter
  tolerance = cv_prosObj$tolerance
  random_seed = cv_prosObj$random_seed
  fit = pros(X, y, alpha = alpha, lambda = lambda, max_iter = max_iter, tolerance = tolerance, random_seed = random_seed)
  res = predict(fit, X_new)
  return ( res )
}


###
# Test
###
test = function() {
  system("cd ../src; R CMD SHLIB R_interface.cpp")
  dyn.load("../src/R_interface.so")

  X_train = data.matrix(read.csv("../../data/X_train.csv", header = F))
  y_train = data.matrix(read.csv("../../data/y_train.csv", header = F))

  X_test = data.matrix(read.csv("../../data/X_test.csv", header = F))
  y_test = data.matrix(read.csv("../../data/y_test.csv", header = F))

  max_iter = 10000
  alpha = c(1, 0, 0, 0, 0, 0)
  lambda = .01
  lambdas = c(.01, .5, 1)
  algorithm = "proximal_gradient_cd"
  K_fold = 10
  tolerance = 10^(-3)
  random_seed = 1032432;

  ###
  # fit test
  ###
  .Call("R_fit", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1),
                 matrix(as.vector(t(alpha)), ncol = 1), as.double(lambda), toString(algorithm),
                 as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  fit = pros(X_train, y_train, lambda = .1)
  fit

  ###
  # predict test
  ###
  .Call("R_predict", matrix(c(1, 1, 1, 1, 1), ncol = 1), as.double(mean(t(y_test))), as.matrix(X_test))
  pred = predict(fit, X_train)
  
  print(mean((y_train - predict(fit, X_train))^2))
  print(mean((y_test - predict(fit, X_test))^2))

  ###
  # CV test
  ###
  .Call("R_cross_validation", as.matrix(X_train), matrix(as.vector(t(y_train)), ncol = 1), as.double(K_fold), 
        matrix(as.vector(t(alpha)), ncol = 1), lambdas, toString(algorithm), 
        as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  cv = cv.pros(X_train, y_train)

  print(mean((y_train - predict(cv, X_train))^2))
  print(mean((y_test - predict(cv, X_test))^2))

}