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

typedef Matrix<double, 6, 1> Vector6d;

struct CVType {
  VectorXd risks;
  vector<double> lambdas;
};

//
// Utils
//
double mean_squared_error(const VectorXd& v, const VectorXd& w) {
  return (v - w).squaredNorm(); 
}

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


// Proximal Gradient
/// =====================================================================================

VectorXd predict(const VectorXd& B, const double intercept, const MatrixXd& X) {
  const int n = X.rows();
  return intercept * VectorXd::Ones(n) + (X * B);
}

//
// Proximal Gradient Coordinate Descent
// Step size: 
// Constant is performing better: 1/((double)max_iter)
// Possible change: Diminishing from Nesterov's Lecture Notes: pow(1 + j, -1/2.0f)
//
VectorXd proximal_gradient_cd(VectorXd B, const MatrixXd& X, const VectorXd& y, 
                              const Vector6d& alpha, const double lambda, 
                              const int max_iter, const double tolerance, const int random_seed) {
  const int n = X.rows();
  const int p = X.cols();

  // Create random permutation for coordinate descent using the Mersenne twister random number engine 64 bit
  vector<int> I(p);
  std::iota (std::begin(I), std::end(I), 0);
  std::mt19937_64 rng(random_seed);

  // Center: Convert X, y to mean 0
  MatrixXd cX = MatrixXd(n, p);
  for (int j = 0; j < X.cols(); j++) {
    cX.col(j) = X.col(j) - (X.col(j).mean() * VectorXd::Ones(n));
  }
  VectorXd cy = y - (y.mean() * VectorXd::Ones(n));

  for (int j = 0; j < max_iter; j++) {
    const VectorXd B_old = B; // Copy B for stopping criterion
    const double h_j = 1/((double)max_iter); // step size

    std::shuffle(std::begin(I), std::end(I), rng); // permute
    for (int& i : I) {
      // derivative of loss + differentiable penalizations
      const double DL_i = -1 * cX.col(i).transpose() * (cy - (cX * B))
                + alpha(1) * lambda * B(i) + alpha(2) * lambda * pow(B(i), 3) 
                + alpha(3) * lambda * pow(B(i), 5) + alpha(4) * lambda * pow(B(i), 7) + alpha(5) * lambda * pow(B(i), 9);

      const double v_i = B(i) - h_j * DL_i; // gradient step

      // Proximal Mapping: Soft Thresholding
      if (v_i < -h_j * alpha(0) * lambda) {
        B(i) = v_i + h_j * alpha(0) * lambda;
      } else if (v_i >= -h_j * alpha(0) * lambda && v_i <= h_j * alpha(0) * lambda) {
        B(i) = 0;
      } else if (v_i > h_j * alpha(0) * lambda) {
        B(i) = v_i - h_j * alpha(0) * lambda;
      }
    }

    // Stop if the norm of the Moreau-Yoshida convolution gradient is less than tolerance
    if ( (1/h_j * (B_old - B)).squaredNorm() < tolerance ) {
      return B;
    }
  }

  std::cout << "Failed to converge!\n";
  return B;
}

// Returns a matrix of B
// We do not sort the lambdas here, they are ordered how you want them
MatrixXd warm_start_proximal_gradient_cd(const MatrixXd& X, const VectorXd& y, 
                                         const Vector6d& alpha, vector<double> lambdas, 
                                         const int max_iter, const double tolerance, const int random_seed) {
  int p = X.cols();
  int L = lambdas.size();
  MatrixXd B_matrix = MatrixXd::Zero(L, p);

  // do the first one normally
  VectorXd B_0 = VectorXd::Zero(p);
  B_matrix.row(0) = proximal_gradient_cd(B_0, X, y, alpha, lambdas[0], max_iter, tolerance, random_seed);

  // Warm start after the first one
  for (int l = 1; l < L; l++) {
    VectorXd B_warm = B_matrix.row((l - 1)); // warm start
    B_matrix.row(l) = proximal_gradient_cd(B_warm, X, y, alpha, lambdas[l], max_iter, tolerance, random_seed);
  }

  return B_matrix;
}

// Prox Gradient Cross Validation
CVType cross_validation_proximal_gradient_cd(const MatrixXd& X, const VectorXd& y, 
                                             const double K, const Vector6d& alpha, vector<double> lambdas, 
                                             const int max_iter, const double tolerance, const int random_seed) {
  int n = X.rows();
  int p = X.cols();
  int L = lambdas.size();
  MatrixXd test_risks_matrix = MatrixXd::Zero(L, K);
  sort(lambdas.begin(), lambdas.end(), std::greater<double>()); // sort the lambdas in place descending

  // Create random permutation using the Mersenne twister random number engine 64 bit
  vector<int> I(n);
  std::iota (std::begin(I), std::end(I), 0);
  std::mt19937_64 rng(random_seed);
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
    MatrixXd B_matrix = warm_start_proximal_gradient_cd(X_train, y_train, alpha, lambdas, max_iter, tolerance, random_seed);
    for (int l = 0; l < L; l++) {
      VectorXd B = B_matrix.row(l).transpose();

      // compute the intercept
      int n = X_train.rows();
      double intercept = 1/((double)n) *  VectorXd::Ones(n).transpose() * (y_train.mean() * VectorXd::Ones(n) - (X_train * B));

      test_risks_matrix(l, k) = mean_squared_error(y_test, predict(B, intercept, X_test));
    }
  }

  // build return
  CVType cv;
  cv.risks = test_risks_matrix.rowwise().mean();
  cv.lambdas = lambdas;
  return cv;
}






















/// DEPRECATED!!!!!
// Subgradient Coordinate Descent
/// =====================================================================================

double logexpsum(const VectorXd& v) {
  double s = 0.0;
  for (int i = 0; i < v.rows(); i++) {
    s  = s + exp(v(i));
  }
  return log(s);
}

double sign(const double x) {
  if (x < 0) { return -1.0; }
  else if (x == 0) { return 0.0; }
  else { return +1.0; }
}

VectorXd sign(VectorXd v) {
  for (int i = 0; i < v.rows(); i++) {
    if (v(i) < 0) { 
      v(i) = -1.0;
    } else if (v(i) > 0) { 
      v(i) = +1.0;
    } else { 
      v(i) = 0.0;
    }
  }
  return v;
}

//
// Subgradient coordinate descent
// Step size: From Nesterov's Lecture Notes
//
VectorXd subgrad_cd(VectorXd B, const MatrixXd& X, const VectorXd& y, const Vector6d& alpha, const double lambda) {
  const int n = X.rows();
  const int p = X.cols();
  const int max_iter = 10000;
  const double tolerance = pow(10, -7);

  // Create random permutation for coordinate descent
  vector<int> I(p);
  std::iota (std::begin(I), std::end(I), 0);
  auto rng = std::default_random_engine {};

  // Copy a Standardize X and y
  MatrixXd cX = MatrixXd(n, p);
  for (int j = 0; j < X.cols(); j++) {
    cX.col(j) = (X.col(j) - X.col(j).mean() * VectorXd::Ones(n));
  }
  VectorXd cy = y - y.mean() * VectorXd::Ones(n);

  for (int j = 0; j < max_iter; j++) {
    const double s = pow(10, (-(log(n)/log(10)))) * pow(1 + j, -1/2.0f); // step size (Chosen using Nesterov's Lecture Notes)
    VectorXd G_j = VectorXd::Ones(p); // Current Subgradient
    
    std::shuffle(std::begin(I), std::end(I), rng); // permute
    for (int& i : I) {
      double Bi_current = B(i);
      VectorXd cX_i = cX.col(i);
      // derivative of loss + penalization
      double DL_i = -1 * cX_i.transpose() * (cy - (cX * B)) 
                  + alpha(1) * lambda * Bi_current + alpha(2) * lambda * pow(Bi_current, 3) 
                  + alpha(3) * lambda * pow(Bi_current, 5) + alpha(4) * lambda * pow(Bi_current, 7) + alpha(5) * lambda * pow(Bi_current, 9);

      // handle l1 subgradient cases and update
      double g_l1_B_i = sign(Bi_current); // subgrad of l1
      if (g_l1_B_i != 0) {
        G_j(i) = DL_i + alpha(0) * lambda * g_l1_B_i;
        B(i) = Bi_current - s * G_j(i);
      } else if (g_l1_B_i == 0) {
        if (DL_i + alpha(0) * lambda < 0) {
          G_j(i) = DL_i + alpha(0) * lambda;
          B(i) = Bi_current - s * G_j(i);
        } else if (DL_i - alpha(0) * lambda > 0) {
          G_j(i) = DL_i - alpha(0) * lambda;
          B(i) = Bi_current - s * G_j(i);
        } else if (DL_i >= -alpha(0) * lambda && DL_i <= alpha(0) * lambda) {
          G_j(i) = DL_i;
          B(i) = 0;
        }
      }
    }

    // Stop if subgradient norm squared is small
    if (G_j.squaredNorm() < tolerance) {
      return B;
    }
  }

  std::cout << "Failed to converge!\n";
  return B;
}

// Returns a matrix of B
// We do not sort the lambdas here, they are ordered how you want them
MatrixXd warm_start_subgrad_cd(const MatrixXd& X, const VectorXd& y, const Vector6d& alpha, vector<double> lambdas) {
  int p = X.cols();
  int L = lambdas.size();
  MatrixXd B_matrix = MatrixXd::Zero(L, p);

  // do the first one normally
  VectorXd B_0 = VectorXd::Zero(p);
  B_matrix.row(0) = subgrad_cd(B_0, X, y, alpha, lambdas[0]);

  // Warm start after the first one
  for (int l = 1; l < L; l++) {
    VectorXd B_warm = B_matrix.row((l - 1)); // warm start
    B_matrix.row(l) = subgrad_cd(B_warm, X, y, alpha, lambdas[l]);
  }

  return B_matrix;
}

// Subgradient Cross Validation
CVType cross_validation_subgrad_cd(const MatrixXd& X, const VectorXd& y, const double K, const Vector6d& alpha, vector<double> lambdas) {
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
    MatrixXd B_matrix = warm_start_subgrad_cd(X_train, y_train, alpha, lambdas);
    for (int l = 0; l < L; l++) {
      VectorXd B = B_matrix.row(l).transpose();

      // compute the intercept
      int n = X_train.rows();
      double intercept = 1/((double)n) *  VectorXd::Ones(n).transpose() * (y_train.mean() * VectorXd::Ones(n) - (X_train * B));
      
      test_risks_matrix(l, k) = mean_squared_error(y_test, predict(B, intercept, X_test));
    }
  }

  // build return
  CVType cv;
  cv.risks = test_risks_matrix.rowwise().mean();
  cv.lambdas = lambdas;
  return cv;
}
