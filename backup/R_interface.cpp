// Libraries here ORDER MATTERS
#include <iostream>

// R stuff here ORDER MATTERS
#include <R.h>
#include <Rinternals.h>

// This allows visibility
extern "C" {

SEXP R_foo(){
  //
  // Copy to R
  //
  SEXP result;
  PROTECT(result = Rf_allocVector(REALSXP, 10));
  GetRNGstate();

  // Copy
  for (int i = 0; i < 10; i++) {
    REAL(result)[i] = i;
  }

  PutRNGstate();
  UNPROTECT(1);
  return result;
}

}