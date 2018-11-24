#include <time.h>
#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

void bench() {
    MatrixXf a = MatrixXf::Random(5000, 5000);
    MatrixXf b = MatrixXf::Random(5000, 5000);
    time_t start = clock();
    MatrixXf c = a * b;
    cout << (double)(clock() - start) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
}

/*
clang++ -std=c++17 -Ieigen test.cpp -o test-O3 -march=native -mfpmath=sse && ./test
*/
int main() {
  bench();
}

