#include "pros.hpp"

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
clang++ -Wall -Wextra -std=c++17 -I ./eigen/ pros_test.cpp -o pros_test && ./pros_test
*/
void test() {
  //std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1); // set precision

  MatrixXd X_train = load_csv<MatrixXd>("X_train.csv");
  VectorXd y_train = load_csv<MatrixXd>("y_train.csv");
  MatrixXd X_test = load_csv<MatrixXd>("X_test.csv");
  VectorXd y_test = load_csv<MatrixXd>("y_test.csv");
  VectorXd B_0 = VectorXd::Zero(X_train.cols());

  int K_fold = 5;
  double lambda = .1;
  double intercept = y_train.mean();

  // create alpha
  double d[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  Map<Vector7d> alpha(d);

  // create lambdas
  vector<double> lambdas;
  lambdas.push_back(pow(10, -3));
  for (int i = 1; i < 10; i++) {
    lambdas.push_back(lambdas[i - 1] + .1);
  }
  // sort(lambdas.begin(), lambdas.end(), std::greater<double>()); // sort in place descending


  // -----------------
  // Proximal Gradient Testing
  // -----------------

  //
  // Single fit test
  //
  VectorXd B_prox = proximal_gradient_cd(B_0, X_train, y_train, alpha, lambda);
  cout << "\n" << B_prox << "\n";
  cout << mean_squared_error(y_train, predict(B_prox, intercept, X_train)) << "\n";

  /*
  //
  // Warm start test
  //
  MatrixXd B_prox_matrix = warm_start_proximal_gradient_cd(X_train, y_train, alpha, lambdas);
  cout << B_prox_matrix.row(B_prox_matrix.rows() - 1).transpose() << "\n";

  //
  // CV test
  //
  CVType cv_prox = cross_validation_proximal_gradient_cd(X_train, y_train, K_fold, alpha, lambdas);
  cout << cv_prox.risks << "\n";

  for (auto& lambda : cv_prox.lambdas) {
    cout << lambda << " ";
  }

  // get the best lambda
  MatrixXf::Index min_row;
  double prox_min = cv_prox.risks.minCoeff(&min_row);
  double prox_best_lambda = cv_prox.lambdas[min_row];
  */

  // -----------------
  // Subgradient Testing
  // -----------------

  //
  // Single fit test
  //
  VectorXd B = subgrad_cd(B_0, X_train, y_train, alpha, lambda);
  cout << "\n" << B << "\n";
  cout << mean_squared_error(y_train, predict(B, intercept, X_train)) << "\n";
  
  /*
  //
  // Warm start test
  //
  MatrixXd B_matrix = warm_start_subgrad_cd(X_train, y_train, alpha, lambdas);
  cout << B_matrix.row(B_matrix.rows() - 1).transpose() << "\n";

  //
  // CV test
  //
  CVType cv = cross_validation_subgrad_cd(X_train, y_train, K_fold, alpha, lambdas);
  cout << cv.risks << "\n";

  for (auto& lambda : cv.lambdas) {
    cout << lambda << " ";
  }

  // get the best lambda
  MatrixXf::Index min_row;
  double min = cv.risks.minCoeff(&min_row);
  double best_lambda = cv.lambdas[min_row];
  */
}

int main() {
  test();
}

