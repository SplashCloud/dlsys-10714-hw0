#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float *m1, const float *m2, float *res,
        size_t p, size_t q, size_t t) {
  // m1: (p x q)
  // m2: (q x t)
  // res: (p x t)
  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < t; j++) {
      float ele = 0.0;
      for (size_t u = 0; u < q; u++) {
        ele += m1[i*q+u] * m2[u*t+j];
      }
      res[i*t+j] = ele;
    }
  }
}

void normalize(float *m, size_t p, size_t q) {
  for (size_t i = 0; i < p; i++) {
    float sum = 0.0;
    for (size_t j = 0; j < q; j++) {
      m[i*q+j] = exp(m[i*q+j]);
      sum += m[i*q+j];
    }
    for (size_t j = 0; j < q; j++) {
      m[i*q+j] /= sum;
    }
  }
}

void transpose(const float *m, float *res, size_t p, size_t q) {
  for (size_t i = 0; i < q; i++) {
    for (size_t j = 0; j < p; j++) {
      res[i*p+j] = m[j*q+i];
    }
  }
}

void minus(float *a, const float *b, size_t p, size_t q) {
  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < q; j++) {
      a[i*q+j] -= b[i*q+j];
    }
  }
}

void multiple(float *a, const float alpha, size_t p, size_t q) {
  for (size_t i = 0; i < p; i++) {
    for (size_t j = 0; j < q; j++) {
      a[i*q+j] *= alpha;
    }
  }
}

void PrintMatrix(const float *m, size_t p, size_t q) {
  std::cout << "matrix(" << p << " x " << q << ")" << std::endl;
  for (size_t i = 0; i < p; i++) {
      for (size_t j = 0; j < q; j++) {
          std::cout << "\t" << m[i*q+j] << ", ";
      }
      std::cout << std::endl;
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t total = m;
    size_t itr = 1;
    while (total > 0) {
      size_t p = batch;
      if (total >= batch) {
        total -= batch;
      } else {
        p = total;
        total = 0;
      }
      float *X0 = new float[p * n]; // (p x n)
      unsigned char *y0 = new unsigned char[p]; // (p)
      size_t offset = (itr-1)*batch;
      size_t end = itr*batch > m ? m : itr*batch;
      for (size_t i = offset; i < end; i++) {
        for (size_t j = 0; j < n; j++) {
          X0[(i-offset)*n+j] = X[i*n+j];
        }
        y0[i-offset] = y[i];
      }
      // Z = X0 * theta
      float *Z = new float[p * k];
      matmul(X0, theta, Z, p, n, k);
      // normalize Z
      normalize(Z, p, k);
      // Z - I
      float *I = new float[p * k]{0.0};
      for (size_t i = 0; i < p; i++) {
        I[i*k+y0[i]] = 1;
      }
      minus(Z, I, p, k);
      // transpose X
      float *X0_T = new float[n * p];
      transpose(X0, X0_T, p, n);
      // matmul
      float *result = new float[n * k];
      matmul(X0_T, Z, result, n, p, k);
      // update
      multiple(result, lr / p, n, k);
      minus(theta, result, n, k);

      delete [] X0;
      delete [] y0;
      delete [] Z;
      delete [] I;
      delete [] X0_T;
      delete [] result;

      itr += 1;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
