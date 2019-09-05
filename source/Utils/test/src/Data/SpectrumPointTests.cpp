#include <Utils/Data/SpectrumPoint.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(SpectrumPointTests, constructor_shouldSetTheAttributes)
{
    constexpr double Frequency = 1;
    constexpr double Amplitude = 2;

    SpectrumPoint spectrumPoint(Frequency, Amplitude);

    EXPECT_EQ(spectrumPoint.frequency(), Frequency);
    EXPECT_EQ(spectrumPoint.amplitude(), Amplitude);
}
