#include <SignalProcessing/Analysis/RealtimeSpectrumAnalyser.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(RealtimeSpectrumAnalyserTests, analyse_shouldReturnTheSpectrumOfEachInput)
{
    constexpr size_t fftSize = 4;
    constexpr size_t sampleFrequency = 48000;
    constexpr size_t channelCount = 1;

    RealtimeSpectrumAnalyser analyser(fftSize, sampleFrequency, channelCount);

    analyser.writePartialData([](size_t i, float* b)
    {
        b[fftSize * i] = 0;
        b[fftSize * i + 1] = 1;
        b[fftSize * i + 2] = 2;
        b[fftSize * i + 3] = 3;
    });
    /*
    analyser.writePartialData([](size_t i, float* b)
    {
        b[fftSize * i] = 0;
        b[fftSize * i + 1] = 1;
        b[fftSize * i + 2] = 2;
        b[fftSize * i + 3] = 3;
        b[fftSize * i + 4] = 4;
    });*/

    analyser.finishWriting();
    analyser.analyse().print();
}
