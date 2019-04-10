#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

using namespace adaptone;
using namespace std;

TEST(CudaSignalProcessorTests, performance)
{
    constexpr size_t frameSampleCount = 32;
    constexpr size_t sampleFrequency = 48000;
    constexpr size_t inputChannelCount = 16;
    constexpr size_t outputChannelCount = 16;
    constexpr PcmAudioFrame::Format inputFormat = PcmAudioFrame::Format::SignedPadded24;
    constexpr PcmAudioFrame::Format outputFormat = PcmAudioFrame::Format::SignedPadded24;
    constexpr size_t parametricEqFilterCount = 2;
    vector<double> frequencies{ 20, 50, 125 };
    constexpr size_t soundLevelLength = 2;
    shared_ptr<AnalysisDispatcher> analysisDispatcher;

    CudaSignalProcessor<float> processor(frameSampleCount,
        sampleFrequency,
        inputChannelCount,
        outputChannelCount,
        inputFormat,
        outputFormat,
        parametricEqFilterCount,
        frequencies,
        soundLevelLength,
        analysisDispatcher);

    PcmAudioFrame inputFrame(outputFormat, outputChannelCount, frameSampleCount);


    constexpr size_t Count = 10000;

    auto start = chrono::system_clock::now();
    for (size_t i = 0; i < Count; i++)
    {
        for (size_t j = 0; j < inputFrame.size(); j++)
        {
            inputFrame[j] = 10;
        }
        PcmAudioFrame outputFrame = processor.process(inputFrame);
    }
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsedSeconds = end - start;

    cout << "Elapsed time = " << elapsedSeconds.count() / Count << " s" << endl;
}

//TODO Faire un test où les paramètres changent un après l'autre.
