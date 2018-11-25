#include "sgcd.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

using std::cout;

//
// Test
//

// CV parser
template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

/*
clang++ -Wall -Wextra -std=c++17 -I ./eigen/ sgcd_test.cpp -o sgcd_test && ./sgcd_test
*/
void test() {
  //std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1); // set precision

  MatrixXd X_train = load_csv<MatrixXd>("X_train.csv");
  VectorXd y_train = load_csv<MatrixXd>("y_train.csv");
  MatrixXd X_test = load_csv<MatrixXd>("X_test.csv");
  VectorXd y_test = load_csv<MatrixXd>("y_test.csv");

  int K_fold = 5;

  // create lambdas
  vector<double> lambdas;
  lambdas.push_back(pow(10, -3));
  for (int i = 1; i < 10; i++) {
    lambdas.push_back(lambdas[i - 1] + .1);
  }
  // sort(lambdas.begin(), lambdas.end(), std::greater<double>()); // sort in place descending

  //
  // Warm start B matrix test
  /*
  MatrixXd B_matrix = warm_start_B_matrix(X_train, y_train, lambdas);
  cout << B_matrix;
  */

  //
  // CV test
  //
  /*
  CVType cv = cross_validation(X_train, y_train, K_fold, lambdas);
  cout << cv.risks << "\n";

  for (auto& lambda : cv.lambdas) {
    cout << lambda << " ";
  }

  // get the best lambda
  MatrixXf::Index min_row;
  double min = cv.risks.minCoeff(&min_row);
  double best_lambda = cv.lambdas[min_row];
  */
  
  //
  // Single fit test
  //
  double lambda = .1;
  double intercept = y_train.mean();

  VectorXd B_0 = VectorXd::Zero(X_train.cols());
  VectorXd B = sparsify(sgcd(B_0, X_train, y_train, lambda), .01);
  cout << "\n" << B << "\n";
  // cout << mean_squared_error(y_train, predict(B, intercept, X_train));
}

int main() {
  test();
}

