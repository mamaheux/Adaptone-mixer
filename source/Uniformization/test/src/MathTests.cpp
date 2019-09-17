//
// Created by pascal on 9/17/19.
//
#include <Uniformization/Math.h>
#include <cstddef>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(MathTests, logSinChirp_shouldReturnTheLogSinChirp)
{

    float T = 1;
    uint32_t Fs = 44100;
    arma::fvec chirp = logSinChirp<arma::fvec>(20.0, 10000.0, T, Fs);

    EXPECT_EQ(chirp.n_elem, 44100);
    float tol = 0.001;
    EXPECT_NEAR(chirp(0), 0.0, tol);
    EXPECT_NEAR(chirp(200), 0.54637, tol);
    EXPECT_NEAR(chirp(5001), 0.96303, tol);
    EXPECT_NEAR(chirp(20000), -0.93309, tol );
    EXPECT_NEAR(chirp(44099), -0.61934, tol);
}
