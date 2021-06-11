#include <cassert>
#include <complex>
#include <random>
#include <iostream>
#include "gcem.hpp"

typedef std::complex<double> complex_f64_t;

constexpr complex_f64_t W_gen(std::size_t N) {
    double Sr = 0.;
    double Si = 0.;
    for (std::size_t k = 0; k < GCEM_EXP_MAX_ITER_SMALL; ++k) {
        double factor = gcem::pow(-2. * GCEM_PI / static_cast<double>(N), k) / static_cast<double>(gcem::factorial(k));
        switch(k % 4) {
            case 0:
                Sr += factor;
                break;
            case 1:
                Si += factor;
                break;
            case 2:
                Sr -= factor;
                break;
            case 3:
                Si -= factor;
                break;

        }
    } 
    return complex_f64_t(Sr, Si);
}

template<std::size_t N_ln2>
complex_f64_t fft_s(complex_f64_t* src, std::size_t k, std::size_t offset) {
    constexpr std::size_t N = 1 << N_ln2;
    constexpr complex_f64_t W = W_gen(N);
    
   if constexpr (N_ln2 == 0) {
       return src[offset];
   } else {
       return fft_s<N_ln2 - 1>(src, k, 2 * offset) + std::pow(W, k) * fft_s<N_ln2 - 1>(src, k, 2 * offset + 1);
   }
}

constexpr std::size_t N_LN = 3;

void fft(complex_f64_t* src, complex_f64_t* dst) {
    constexpr std::size_t N = 1 << N_LN;
    for (std::size_t k = 0; k < N; ++k)
        dst[k] = fft_s<N_LN>(src, k, 0);
}


int main(void) {
    constexpr std::size_t N = 1 << N_LN;
    complex_f64_t* A = static_cast<complex_f64_t*>(malloc(N * sizeof(complex_f64_t)));
    complex_f64_t* B = static_cast<complex_f64_t*>(malloc(N * sizeof(complex_f64_t)));
   
    std::random_device r; 
    std::uniform_real_distribution<double> unif(-1., 1.);
    std::default_random_engine re(r());
    
    for (std::size_t i = 0; i < N; ++i) {
        complex_f64_t a(unif(re), 0);
        A[i] = a;
    }
    fft(A, B);
    
    complex_f64_t sum_A(0, 0);
    complex_f64_t sum_B(0, 0);

    for (std::size_t i = 0; i < N; ++i) {
        std::cout << B[i] << "\n";
        sum_A += A[i];
        sum_B += B[i];
    }
    sum_A /= static_cast<double>(N);
    sum_B /= static_cast<double>(N);
    std::cout << "Input mean: " << sum_A << " Output mean: " << sum_B << "Diff: " << sum_A / sum_B << "\n";

    free(A);
    free(B);
    return 0;
}
