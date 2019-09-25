#include <Uniformization/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(MathTests, logSinChirp_shouldReturnTheLogSinChirp)
{
    constexpr float T = 1;
    constexpr uint32_t Fs = 44100;
    constexpr float Tol = 0.001;

    arma::fvec chirp = logSinChirp<arma::fvec>(20.0, 10000.0, T, Fs);

    EXPECT_EQ(chirp.n_elem, 44100);
    EXPECT_NEAR(chirp(0), 0.0, Tol);
    EXPECT_NEAR(chirp(200), 0.54637, Tol);
    EXPECT_NEAR(chirp(5001), 0.96303, Tol);
    EXPECT_NEAR(chirp(20000), -0.93309, Tol);
    EXPECT_NEAR(chirp(44099), -0.61934, Tol);
}
