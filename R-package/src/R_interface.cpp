// Libraries here ORDER MATTERS
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "sgcd.hpp"

// R stuff here ORDER MATTERS
#include <R.h>
#include <Rinternals.h>

using namespace Eigen;
using std::vector;

// This allows visibility
extern "C" {

SEXP R_fit(SEXP X_, SEXP y_, SEXP lambda_){
  SEXP result;
  GetRNGstate();

  // Handle X
  SEXP dim_X = getAttrib(X_, R_DimSymbol);
  const int nrow_X = INTEGER(dim_X)[0];
  const int ncol_X = INTEGER(dim_X)[1];
  double* p_X = REAL(X_); // pointer

  // Handle y
  SEXP dim_y = getAttrib(y_, R_DimSymbol);
  const int nrow_y = INTEGER(dim_y)[0];
  double* p_y = REAL(y_); // pointer

  // Setup
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(p_X, nrow_X, ncol_X);
  Map<VectorXd> y(p_y, nrow_y);
  double lambda = REAL(lambda_)[0];

  // fit
  VectorXd B_0 = VectorXd::Zero(X.cols());
  VectorXd B = sparsify(sgcd(B_0, X, y, lambda));

  //
  // Copy to R
  //
  PROTECT(result = Rf_allocVector(REALSXP, B.rows()));

  // Copy
  for (int i = 0; i < B.rows(); i++) {
    REAL(result)[i] = B(i);
  }

  PutRNGstate();
  UNPROTECT(1);
  return result;
}

SEXP R_predict(SEXP X_, SEXP B_, SEXP intercept_){
  SEXP result;
  GetRNGstate();

  // Handle X
  SEXP dim_X = getAttrib(X_, R_DimSymbol) ;
  const int nrow_X = INTEGER(dim_X)[0];
  const int ncol_X = INTEGER(dim_X)[1];
  double* p_X = REAL(X_); // pointer

  // Handle B
  SEXP dim_B = getAttrib(B_, R_DimSymbol);
  const int nrow_B = INTEGER(dim_B)[0];
  double* p_B = REAL(B_); // pointer

  // Setup
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(p_X, nrow_X, ncol_X);
  Map<VectorXd> B(p_B, nrow_B);
  double intercept = REAL(intercept_)[0];

  // Predict
  VectorXd pred = predict(X, B, intercept);

  //
  // Copy to R
  //
  PROTECT(result = Rf_allocVector(REALSXP, pred.rows()));

  // Copy
  for (int i = 0; i < pred.rows(); i++) {
    REAL(result)[i] = pred(i);
  }

  PutRNGstate();
  UNPROTECT(1);
  return result;
}

SEXP R_cross_validation(SEXP X_, SEXP y_, SEXP K_fold_, SEXP lambdas_){  
  SEXP res;
  GetRNGstate();

  // Handle X
  SEXP dim_X = getAttrib(X_, R_DimSymbol);
  const int nrow_X = INTEGER(dim_X)[0];
  const int ncol_X = INTEGER(dim_X)[1];
  double* p_X = REAL(X_); // pointer

  // Handle y
  SEXP dim_y = getAttrib(y_, R_DimSymbol);
  const int nrow_y = INTEGER(dim_y)[0];
  double* p_y = REAL(y_); // pointer

  // Handle lambdas
  const int L = Rf_length(lambdas_);

  // Setup
  Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(p_X, nrow_X, ncol_X);
  Map<VectorXd> y(p_y, nrow_y);
  double K_fold = REAL(K_fold_)[0];

  // Build lambdas
  vector<double> lambdas;
  lambdas.assign(REAL(lambdas_), REAL(lambdas_) + L);

  // CV
  CVType cv = cross_validation(X, y, K_fold, lambdas);
  vector<double> cv_lambdas = cv.lambdas;
  VectorXd cv_risks = cv.risks;

  // get location of minimum
  MatrixXf::Index min_row;
  cv.risks.minCoeff(&min_row);
  double best_lambda = cv.lambdas[min_row];

  //
  // Copy to R
  //
  const char *names[] = {"best_lambda", "lambdas", "risks", ""};
  res = PROTECT(mkNamed(VECSXP, names)); // create response

  SET_VECTOR_ELT(res, 0, ScalarReal(best_lambda));

  // Copy lambdas
  SEXP res_lambdas;
  PROTECT(res_lambdas = Rf_allocVector(REALSXP, cv_lambdas.size()));
  for (size_t i = 0; i < cv_lambdas.size(); i++) {
    REAL(res_lambdas)[i] = cv_lambdas[i];
  }
  SET_VECTOR_ELT(res, 1, res_lambdas);

  // Copy Risks
  SEXP res_risks;
  PROTECT(res_risks = Rf_allocVector(REALSXP, cv_risks.rows()));
  for (int i = 0; i < cv_risks.rows(); i++) {
    REAL(res_risks)[i] = cv_risks(i);
  }
  SET_VECTOR_ELT(res, 2, res_risks);

  PutRNGstate();
  UNPROTECT(3);
  return res;
}

}

