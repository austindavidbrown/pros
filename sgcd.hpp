#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

#include <iostream>

using namespace Eigen;
using std::vector;
using std::array;
using std::sort;
using std::cout;

// long double matrix, vector, scalar
typedef long double ld;
typedef Matrix<long double, Dynamic, Dynamic> MatrixXld;
typedef Matrix<long double, Dynamic, 1> VectorXld;

//
// Optimization
//
double sign(double x) {
  if (x < 0) { return -1.0; }
  else if (x == 0) { return 0.0; }
  else { return +1.0; }
}

// TODO stopping criterion
// TODO step size s = 1/(10**3 + j) # s = 1/(10**7 + j)**(1/2)
// TODO add elasticnet
VectorXd sgcd(VectorXd B, const MatrixXd& X, const VectorXd& y, const double lambda) {
  const int n = X.rows();
  const int p = X.cols();
  const int max_iter = 100000;
  const double tolerance = pow(10, -7);

  // Create random permutation for coordinate descent
  vector<int> I(p);
  std::iota (std::begin(I), std::end(I), 0);
  auto rng = std::default_random_engine {};

  // Center
  MatrixXd C = MatrixXd::Identity(n, n) - 1/((double)n) * VectorXd::Ones(n) * VectorXd::Ones(n).transpose();
  MatrixXd cX = C * X;
  VectorXd cy = y - y.mean() * VectorXd::Ones(n);

  for (int j = 0; j < max_iter; j++) {
    const double s = pow(10, (-(log(n)/log(10)))) * pow(1 + j, -1/2.0f); // step size (Chosen using Nesterov's Lecture Notes)
    VectorXd G_j = VectorXd::Ones(p); // Current Subgradient
    std::shuffle(std::begin(I), std::end(I), rng); // permute
    for (int& i : I) {
      double Bi_current = B(i);
      VectorXd cX_i = cX.col(i);
      double DL_i = -1 * cX_i.transpose() * (cy - (cX * B)); // derivative of loss

      // handle l1 subgradient cases and update
      double g_l1_B_i = sign(Bi_current); // subgrad of l1
      if (g_l1_B_i != 0) {
        G_j(i) = DL_i + lambda * g_l1_B_i;
        B(i) = Bi_current - s * G_j(i);
      } else if (g_l1_B_i == 0) {
        if (DL_i < -lambda) {
          G_j(i) = DL_i + lambda;
          B(i) = Bi_current - s * G_j(i);
        } else if (DL_i > lambda) {
          G_j(i) = DL_i - lambda;
          B(i) = Bi_current - s * G_j(i);
        } else if (DL_i >= -lambda && DL_i <= lambda) {
          G_j(i) = DL_i;
          B(i) = 0;
        }
      }
    }

    // stop if subgradient norm squared is small
    if (G_j.squaredNorm() < tolerance) {
      return B;
    }
  }

  std::cout << "Failed to converge!\n";
  return B;
}

VectorXd predict(const VectorXd& B, const double& intercept, const MatrixXd& X) {
  int n = X.rows();
  return intercept * VectorXd::Ones(n) + (X * B);
}

VectorXd sparsify(VectorXd B, double tolerance) {
  for (int i = 0; i < B.rows(); i++) {
    if (B(i) < tolerance) {
      B(i) = 0.0f;
    }
  }
  return B;
}

//
// Cross Validation
//

// from Yury Stackoverflow
template<typename T>
vector<vector<T>> partition(const vector<T>& S, size_t n) {
  vector<vector<T>> partitions;

  size_t length = S.size() / n;
  size_t remainder = S.size() % n;

  size_t begin = 0;
  size_t end = 0;
  for (size_t i = 0; i < n; ++i) {
    if (remainder > 0) {
      end += length + !!(remainder--);
    } else {
      end += length;
    }
    partitions.push_back(vector<T>(S.begin() + begin, S.begin() + end));
    begin = end;
  }

  return partitions;
}

double mean_squared_error(const VectorXd& v, const VectorXd& w) {
  return (v - w).squaredNorm(); 
}

// We do not sort the lambdas here, they are ordered how you want them
MatrixXd warm_start_B_matrix(const MatrixXd& X, const VectorXd& y, vector<double> lambdas) {
  int p = X.cols();
  int L = lambdas.size();
  MatrixXd B_matrix = MatrixXd::Zero(L, p);

  // do the first one normally
  VectorXd B_0 = VectorXd::Zero(p);
  B_matrix.row(0) = sgcd(B_0, X, y, lambdas[0]);

  // Warm start after the first one
  for (int l = 1; l < L; l++) {
    VectorXd B_warm = B_matrix.row((l - 1)); // warm start
    B_matrix.row(l) = sgcd(B_warm, X, y, lambdas[l]);
  }

  return B_matrix;
}

struct CVType {
  VectorXd risks;
  vector<double> lambdas;
};

// TODO I only need to order the lambdas once
CVType cross_validation(const MatrixXd& X, const VectorXd& y, const double K, vector<double> lambdas) {
  int n = X.rows();
  int p = X.cols();
  int L = lambdas.size();
  MatrixXd test_risks_matrix = MatrixXd::Zero(L, K);
  sort(lambdas.begin(), lambdas.end(), std::greater<double>()); // sort the lambdas in place descending

  // Create random permutation
  vector<int> I(n);
  std::iota (std::begin(I), std::end(I), 0);
  auto rng = std::default_random_engine {};
  std::shuffle(std::begin(I), std::end(I), rng); // permute

  vector<vector<int>> partitions = partition(I, K);
  for (size_t k = 0; k < partitions.size(); k++) {
    vector<int> TEST = partitions[k];

    // Build training indices
    vector<int> TRAIN;
    for (int& i : I) {
      bool exists = false;
      for (int& j : TEST) {
        if (j == i) {
          exists = true;
        }
      }
      if (exists == false) {
        TRAIN.push_back(i);
      }
    }

    // Build X_train, y_train
    MatrixXd X_train = MatrixXd(TRAIN.size(), p);
    VectorXd y_train = VectorXd(TRAIN.size());
    for (size_t i = 0; i < TRAIN.size(); i++) {
      X_train.row(i) = X.row(TRAIN[i]);
      y_train.row(i) = y.row(TRAIN[i]);
    }

    // Build X_test, y_test
    MatrixXd X_test = MatrixXd(TEST.size(), p);
    VectorXd y_test = VectorXd(TEST.size());
    for (size_t i = 0; i < TEST.size(); i++) {
      X_test.row(i) = X.row(TEST[i]);
      y_test.row(i) = y.row(TEST[i]);
    }

    // do the computation
    double intercept = y_train.mean();
    MatrixXd B_matrix = warm_start_B_matrix(X_train, y_train, lambdas);
    for (int l = 0; l < L; l++) {
      VectorXd B = B_matrix.row(l).transpose();
      test_risks_matrix(l, k) = mean_squared_error(y_test, predict(B, intercept, X_test));
    }
  }

  // build return
  CVType cv;
  cv.risks = test_risks_matrix.rowwise().mean();
  cv.lambdas = lambdas;
  return cv;
}
