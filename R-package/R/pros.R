#' Pros
#' 
#' The \code{pros} function is used to fit a single regression model with a specified penalization. 
#' 
#' @param X is an \eqn{n \times m}-dimensional matrix of the data.
#' @param y is an \eqn{n \times m}-dimensional matrix of the data.
#' @param alpha is a \eqn{6}-dimensional vector of the convex combination corresponding to the penalization:
#' \itemize{
#'   \item \eqn{\alpha_1} is the \eqn{l^1} penalty.
#'   \item \eqn{\alpha_2} is the \eqn{l^2} penalty.
#'   \item \eqn{\alpha_3} is the \eqn{l^4} penalty.
#'   \item \eqn{\alpha_4} is the \eqn{l^6} penalty.
#'   \item \eqn{\alpha_5} is the \eqn{l^8} penalty.
#'   \item \eqn{\alpha_6} is the \eqn{l^10} penalty.
#' }
#' @param lambda is the Lagrangian dual penalization parameter.
#' @param step_size is a tuning parameter defining the step size. Larger values are more aggressive and smaller values are less aggressive.
#' @param algorithm is the optimization algorithm 
#' \itemize{
#'   \item \code{proximal_gradient_cd} uses proximal gradient coordinate descent.
#'   \item \code{subgradient_cd} uses subgradient coordinate descent.
#' }
#' @param max_iter is the maximum iterations the algorithm will run regardless of convergence.
#' @param tolerance is the accuracy of the stopping criterion.
#' @param random_seed is the random seed used in the algorithms.
#' 
#' @return 
#' A class \code{pros}
#'
#' @export
pros = function(X, y, 
                alpha = c(1, 0, 0, 0, 0, 0), lambda, step_size,
                algorithm = "proximal_gradient_cd", max_iter = 10000, tolerance = 10^(-8), random_seed = 0) {
  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector

  if (length(alpha) != 6) {
    stop("alpha needs to be length 6")
  }
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  res = .Call("R_fit", as.matrix(X), y, 
            alpha, as.double(lambda), as.double(step_size),
            toString(algorithm), as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  class(res) = "pros"
  return ( res )
}


#' Pros Prediction
#' 
#' The prediction function for \code{pros}.
#' 
#' @param prosObj an object of class \code{pros}
#' @param X is an \eqn{n \times m}-dimensional matrix of the data.
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
#' The \code{cv.pros} function is used for K-fold cross-validation.
#' 
#' @param X is an \eqn{n \times m}-dimensional matrix of the data.
#' @param y is an \eqn{n \times m}-dimensional matrix of the data.
#' @param alpha is a \eqn{6}-dimensional vector of the convex combination corresponding to the penalization:
#' \itemize{
#'   \item \eqn{\alpha_1} is the \eqn{l^1} penalty.
#'   \item \eqn{\alpha_2} is the \eqn{l^2} penalty.
#'   \item \eqn{\alpha_3} is the \eqn{l^4} penalty.
#'   \item \eqn{\alpha_4} is the \eqn{l^6} penalty.
#'   \item \eqn{\alpha_5} is the \eqn{l^8} penalty.
#'   \item \eqn{\alpha_6} is the \eqn{l^10} penalty.
#' }
#' @param lambdas is a vector of dual penalization values to be evaluated.
#' @param step_size is a tuning parameter defining the step size. Larger values are more aggressive and smaller values are less aggressive.
#' @param algorithm is the optimization algorithm 
#' \itemize{
#'   \item \code{proximal_gradient_cd} uses proximal gradient coordinate descent.
#'   \item \code{subgradient_cd} uses subgradient coordinate descent.
#' }
#' @param max_iter is the maximum iterations the algorithm will run regardless of convergence.
#' @param tolerance is the accuracy of the stopping criterion.
#' @param random_seed is the random seed used in the algorithms.
#'
#'
#' @return 
#' A class \code{cv_pros}
#'
#' @export
cv.pros = function(X, y, 
                   K_fold = 10, alpha = c(1, 0, 0, 0, 0, 0), lambdas = c(), step_size,
                   algorithm = "proximal_gradient_cd", max_iter = 10000, tolerance = 10^(-8), random_seed = 0) {

  # Set default lambdas for the user
  # There is no theoretical justification here.
  if (length(lambdas) == 0) {
    lambdas = seq(3/2 * ncol(X), 3 * ncol(X), .1)
  }

  y = matrix(as.vector(t(y)), ncol = 1) # convert to column vector
  alpha = matrix(as.vector(t(alpha)), ncol = 1) # convert to column vector

  res = .Call("R_cross_validation", as.matrix(X), y, 
              as.double(K_fold), alpha, as.vector(lambdas), as.double(step_size),
              toString(algorithm), as.integer(max_iter), as.double(tolerance), as.integer(random_seed))
  res$X = X
  res$y = y
  res$alpha = alpha
  res$step_size = step_size
  res$max_iter = max_iter
  res$tolerance = tolerance
  res$random_seed = random_seed
  class(res) = "cv_pros"
  return ( res )
}

#' Cross-validation Prediction
#' 
#' The prediction function for \code{cv.pros}.
#' 
#' @param cv_prosObj an object of class \code{cv_pros}
#' @param X_new is an \eqn{n \times m}-dimensional matrix of the data.
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
  step_size = cv_prosObj$step_size
  max_iter = cv_prosObj$max_iter
  tolerance = cv_prosObj$tolerance
  random_seed = cv_prosObj$random_seed
  fit = pros(X, y, alpha = alpha, lambda = lambda, step_size = step_size, max_iter = max_iter, tolerance = tolerance, random_seed = random_seed)
  res = predict(fit, X_new)
  return ( res )
}