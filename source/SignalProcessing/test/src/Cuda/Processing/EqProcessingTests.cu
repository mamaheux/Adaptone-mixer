#include <SignalProcessingTests/CudaFreeGuard.h>

#include <SignalProcessing/Cuda/Processing/EqProcessing.h>

#include <Utils/Configuration/Properties.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace adaptone;
using namespace std;

vector<BiquadCoefficients<double>> biquadCoefficientsFromProperties(const Properties& properties, const string& b0Key,
    const string& b1Key, const string& a1Key, const string& a2Key)
{
    vector<double> b0 = properties.get<vector<double>>(b0Key);
    vector<double> b1 = properties.get<vector<double>>(b1Key);
    vector<double> a1 = properties.get<vector<double>>(a1Key);
    vector<double> a2 = properties.get<vector<double>>(a2Key);

    vector<BiquadCoefficients<double>> bc(b0.size());

    for (size_t i = 0; i < bc.size(); i++)
    {
        bc[i].b0 = b0[i];
        bc[i].b1 = b1[i];
        bc[i].b2 = 0;
        bc[i].a1 = a1[i];
        bc[i].a2 = a2[i];
    }

    return bc;
}

void initEqBuffers(CudaEqBuffers<double>& eqBuffers, const vector<BiquadCoefficients<double>>& bc1,
    const vector<BiquadCoefficients<double>>& bc2, const vector<BiquadCoefficients<double>>& bc3,
    const vector<double>& d0)
{
    cudaMemcpy(eqBuffers.biquadCoefficients(0), bc1.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(eqBuffers.biquadCoefficients(1), bc2.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(eqBuffers.biquadCoefficients(2), bc3.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<double>), cudaMemcpyHostToDevice);

    cudaMemcpy(eqBuffers.d0(), d0.data(), d0.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void initEqBuffers(CudaEqBuffers<double>& eqBuffers, const vector<BiquadCoefficients<double>>& bc1,
    const vector<BiquadCoefficients<double>>& bc2, const vector<double>& d0)
{
    cudaMemcpy(eqBuffers.biquadCoefficients(0), bc1.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(eqBuffers.biquadCoefficients(1), bc2.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<double>), cudaMemcpyHostToDevice);

    cudaMemcpy(eqBuffers.d0(), d0.data(), d0.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void mallocFrames(double** inputFrames, double** outputFrame, size_t channelCount, size_t frameCount,
    size_t frameSampleCount)
{
    size_t inputFramesSize = channelCount * frameCount * frameSampleCount * sizeof(double);
    size_t outputFrameSize = channelCount * frameSampleCount * sizeof(double);
    cudaMallocManaged(reinterpret_cast<void**>(inputFrames), inputFramesSize);
    cudaMallocManaged(reinterpret_cast<void**>(outputFrame), outputFrameSize);

    cudaMemset(*inputFrames, 0, inputFramesSize);
    cudaMemset(*outputFrame, 0, outputFrameSize);
}

__global__ void processEqKernel(CudaEqBuffers<double> buffers, double* inputFrames, double* currentOutputFrame,
    size_t currentFrameIndex)
{
    processEq(buffers, inputFrames, currentOutputFrame, currentFrameIndex);
}

TEST(EqProcessingTests, processEq_dirac_shouldGenerateTheRightOutput)
{
    constexpr double MaxAbsErrorFactor = 0.000001;

    constexpr size_t ChannelCount = 3;
    constexpr size_t FrameCount = 2;
    constexpr size_t FrameSampleCount = 32;

    Properties properties("resources/Cuda/Processing/EqProcessingTests/impz_tests.properties");
    vector<BiquadCoefficients<double>> bc1 = biquadCoefficientsFromProperties(properties, "b0_channel1", "b1_channel1",
        "a1_channel1", "a2_channel1");
    vector<BiquadCoefficients<double>> bc2 = biquadCoefficientsFromProperties(properties, "b0_channel2", "b1_channel2",
        "a1_channel2", "a2_channel2");
    vector<BiquadCoefficients<double>> bc3 = biquadCoefficientsFromProperties(properties, "b0_channel3", "b1_channel3",
        "a1_channel3", "a2_channel3");

    ASSERT_EQ(bc1.size(), bc2.size());
    ASSERT_EQ(bc1.size(), bc3.size());

    vector<double> d0{ properties.get<double>("d0_channel1"), properties.get<double>("d0_channel2"),
        properties.get<double>("d0_channel3") };

    CudaEqBuffers<double> eqBuffers(ChannelCount, bc1.size(), FrameCount, FrameSampleCount);
    initEqBuffers(eqBuffers, bc1, bc2, bc3, d0);

    double* inputFrames;
    double* outputFrame;
    mallocFrames(&inputFrames, &outputFrame, ChannelCount, FrameCount, FrameSampleCount);
    CudaFreeGuard inputFramesFreeGuard(inputFrames);
    CudaFreeGuard outputFrameFreeGuard(outputFrame);

    inputFrames[0] = 1;
    inputFrames[FrameSampleCount] = 2;
    inputFrames[2 * FrameSampleCount] = 3;

    vector<vector<double>> impz{ properties.get<vector<double>>("impz_channel1"),
        properties.get<vector<double>>("impz_channel2"), properties.get<vector<double>>("impz_channel3") };

    ASSERT_EQ(impz[0].size(), impz[1].size());
    ASSERT_EQ(impz[0].size(), impz[2].size());

    size_t frameToProcessCount = impz[0].size() / FrameSampleCount;
    size_t currentFrameIndex = 0;
    for (size_t i = 0; i < frameToProcessCount; i++)
    {
        processEqKernel<<<1, 128>>>(eqBuffers, inputFrames, outputFrame, currentFrameIndex);
        cudaDeviceSynchronize();

        for (size_t channelIndex = 0; channelIndex < ChannelCount; channelIndex++)
        {
            for (size_t sampleIndex = 0; sampleIndex < FrameSampleCount; sampleIndex++)
            {
                double expectedValue = impz[channelIndex][i * FrameSampleCount + sampleIndex];
                double value = outputFrame[channelIndex * FrameSampleCount + sampleIndex] / (channelIndex + 1);
                ASSERT_NEAR(expectedValue, value, abs(expectedValue * MaxAbsErrorFactor)) << "frame=" << i <<
                    ", channel=" << channelIndex << ", sample=" << sampleIndex;
            }
        }

        inputFrames[0] = 0;
        inputFrames[FrameSampleCount] = 0;
        inputFrames[2 * FrameSampleCount] = 0;
        currentFrameIndex = (currentFrameIndex + 1) % FrameCount;
    }
}

TEST(EqProcessingTests, processEq_song_shouldGenerateTheRightOutput)
{
    constexpr double MaxAbsErrorFactor = 0.00001;

    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameCount = 2;
    constexpr size_t FrameSampleCount = 32;

    Properties properties("resources/Cuda/Processing/EqProcessingTests/song_tests.properties");
    vector<BiquadCoefficients<double>> bc1 = biquadCoefficientsFromProperties(properties, "b0_channel1", "b1_channel1",
        "a1_channel1", "a2_channel1");
    vector<BiquadCoefficients<double>> bc2 = biquadCoefficientsFromProperties(properties, "b0_channel2", "b1_channel2",
        "a1_channel2", "a2_channel2");

    ASSERT_EQ(bc1.size(), bc2.size());

    vector<double> d0{ properties.get<double>("d0_channel1"), properties.get<double>("d0_channel2") };

    CudaEqBuffers<double> eqBuffers(ChannelCount, bc1.size(), FrameCount, FrameSampleCount);
    initEqBuffers(eqBuffers, bc1, bc2, d0);

    double* inputFrames;
    double* outputFrame;
    mallocFrames(&inputFrames, &outputFrame, ChannelCount, FrameCount, FrameSampleCount);
    CudaFreeGuard inputFramesFreeGuard(inputFrames);
    CudaFreeGuard outputFrameFreeGuard(outputFrame);

    vector<double> x = properties.get<vector<double>>("x");
    vector<vector<double>> y{ properties.get<vector<double>>("y_channel1"),
        properties.get<vector<double>>("y_channel2") };

    size_t frameToProcessCount = x.size() / FrameSampleCount;
    size_t currentFrameIndex = 0;

    for (size_t i = 0; i < frameToProcessCount; i++)
    {
        double* currentInputFrame = inputFrames + currentFrameIndex * ChannelCount * FrameSampleCount;
        memcpy(currentInputFrame, x.data() + i * FrameSampleCount, FrameSampleCount * sizeof(double));
        memcpy(currentInputFrame + FrameSampleCount, x.data() + i * FrameSampleCount,
            FrameSampleCount * sizeof(double));

        processEqKernel<<<1, 64>>>(eqBuffers, inputFrames, outputFrame, currentFrameIndex);
        cudaDeviceSynchronize();

        for (size_t channelIndex = 0; channelIndex < ChannelCount; channelIndex++)
        {
            for (size_t sampleIndex = 0; sampleIndex < FrameSampleCount; sampleIndex++)
            {
                double expectedValue = y[channelIndex][i * FrameSampleCount + sampleIndex];
                double value = outputFrame[channelIndex * FrameSampleCount + sampleIndex];
                ASSERT_NEAR(expectedValue, value, abs(expectedValue * MaxAbsErrorFactor)) << "frame=" << i <<
                    ", channel=" << channelIndex << ", sample=" << sampleIndex;
            }
        }

        currentFrameIndex = (currentFrameIndex + 1) % FrameCount;
    }
}
