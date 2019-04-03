#include <SignalProcessing/Cuda/Processing/EqProcessing.h>

#include <Utils/Configuration/Properties.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

vector<BiquadCoefficients<float>> biquadCoefficientsFromProperties(const Properties& properties,
    const string& b0Key, const string& b1Key, const string& a1Key, const string& a2Key)
{
    vector<float> b0 = properties.get<vector<float>>(b0Key);
    vector<float> b1 = properties.get<vector<float>>(b1Key);
    vector<float> a1 = properties.get<vector<float>>(a1Key);
    vector<float> a2 = properties.get<vector<float>>(a2Key);

    ASSERT_EQ(b0.size(), b1.size());
    ASSERT_EQ(b0.size(), a1.size());
    ASSERT_EQ(b0.size(), a2.size());

    vector<BiquadCoefficients<float>> bc(b0.size());

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

void initEqBuffers(CudaEqBuffers<float>& eqBuffers, const vector<BiquadCoefficients<float>>& bc1,
    const vector<BiquadCoefficients<float>>& bc2, const vector<BiquadCoefficients<float>>& bc3, const vector<float>& d0)
{
    cudaMemcpy(eqBuffers.biquadCoefficients(0), bc1.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(eqBuffers.biquadCoefficients(1), bc2.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(eqBuffers.biquadCoefficients(2), bc3.data(),
        eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<float>), cudaMemcpyHostToDevice);

    cudaMemcpy(eqBuffers.d0(), d0.data(), d0.size() * sizeof(float), cudaMemcpyHostToDevice);
}

voidmallocFrames(float** inputFrames, float** outputFrame, size_t channelCount, size_t frameCount,
    size_t frameSampleCount)
{
    size_t inputFramesSize = channelCount * frameCount * frameSampleCount * sizeof(float);
    size_t outputFrameSize = channelCount * frameCount * frameSampleCount * sizeof(float);
    cudaMallocManaged(reinterpret_cast<void**>(inputFrames), inputFrameSize);
    cudaMallocManaged(reinterpret_cast<void**>(outputFrames), outputFrameSize);

    cudaMemset(inputFrames, 0, inputFramesSize);
    cudaMemset(outputFrame, 0, outputFramesSize);
}

__global__ void processEqKernel(CudaEqBuffers<float> buffers, float* inputFrames, float* currentOutputFrame,
    size_t currentFrameIndex)
{
    processEq(buffers, inputFrames, currentOutputFrame, currentFrameIndex);
}

TEST(EqProcessingTests, processEq_shouldGenerateTheRightImpulseResponse)
{
    constexpr size_t ChannelCount = 3;
    constexpr size_t FrameCount = 2;
    constexpr size_t FrameSampleCount = 32;
    constexpr float MaxAbsErrorFactor = 0.001;

    Properties properties("resources/Cuda/Processing/EqProcessingTests/impz_tests.properties");
    vector<BiquadCoefficients<float>> bc1 = biquadCoefficientsFromProperties(properties, "b0_channel1", "b1_channel1",
        "a1_channel1", "a2_channel1");
    vector<BiquadCoefficients<float>> bc2 = biquadCoefficientsFromProperties(properties, "b0_channel2", "b1_channel2",
        "a1_channel2", "a2_channel2");
    vector<BiquadCoefficients<float>> bc2 = biquadCoefficientsFromProperties(properties, "b0_channel3", "b1_channel3",
        "a1_channel3", "a2_channel3");

    ASSERT_EQ(bc1.size(), bc2.size());
    ASSERT_EQ(bc1.size(), bc3.size());

    vector<float> d0{ properties.get<float>("d0_channel1"), properties.get<float>("d0_channel2"),
        properties.get<float>("d0_channel3") };

    CudaEqBuffers<float> eqBuffers(ChannelCount, bc1.size(), FrameCount, FrameSampleCount);
    initEqBuffers(eqBuffers, bc1, bc2, bc3, d0);

    float* inputFrames;
    float* outputFrame;
    mallocFrames(&inputFrames, &outputFrame);

    inputFrames[0] = 1;
    inputFrames[FrameSampleCount] = 1;
    inputFrames[2 * FrameSampleCount] = 1;

    vector<vector<float>> impz{ properties.get<vector<float>>("impz_channel1"),
        properties.get<vector<float>>("impz_channel2"), properties.get<vector<float>>("impz_channel3") };

    ASSERT_EQ(impz[0].size(), impz[1].size());
    ASSERT_EQ(impz[0].size(), impz[2].size());

    size_t frameToProcessCount = impz1.size() / FrameSampleCount;
    size_t currentFrameIndex = 0;
    for (size_t i = 0; i < frameToProcessCount; i++)
    {
        processEqKernel<<<1, 128>>>(eqBuffers, inputFrames, outputFrame, currentFrameIndex);
        cudaDeviceSynchronize();

        for (size_t channelIndex = 0; channelIndex < ChannelCount; channelIndex++)
        {
            for (size_t sampleIndex = 0; sampleIndex < FrameSampleCount; sampleIndex++)
            {
                float expectedValue = impz[channelIndex][i * FrameSampleCount + sampleIndex];
                float value = outputFrame[channelIndex * FrameSampleCount + sampleIndex];
                ASSERT_NEAR(expectedValue, value, expectedValue * MaxAbsErrorFactor);
            }
        }

        inputFrames[0] = 0;
        inputFrames[FrameSampleCount] = 0;
        inputFrames[2 * FrameSampleCount] = 0;
        currentFrameIndex = (currentFrameIndex + 1) % FrameCount;
    }
}

TEST(EqProcessingTests, processEq_shouldGenerateTheRightSong)
{
    constexpr size_t ChannelCount = 2;
    constexpr size_t FrameCount = 2;
    constexpr size_t FrameSampleCount = 32;
    constexpr float MaxAbsErrorFactor = 0.001;

    Properties properties("resources/Cuda/Processing/EqProcessingTests/song_tests.properties");
    vector<BiquadCoefficients<float>> bc1 = biquadCoefficientsFromProperties(properties, "b0_channel1", "b1_channel1",
        "a1_channel1", "a2_channel1");
    vector<BiquadCoefficients<float>> bc2 = biquadCoefficientsFromProperties(properties, "b0_channel2", "b1_channel2",
        "a1_channel2", "a2_channel2");

    ASSERT_EQ(bc1.size(), bc2.size());

    vector<float> d0{ properties.get<float>("d0_channel1"), properties.get<float>("d0_channel2") };

    CudaEqBuffers<float> eqBuffers(ChannelCount, bc1.size(), FrameCount, FrameSampleCount);
    initEqBuffers(eqBuffers, bc1, bc2, bc3, d0);

    float* inputFrames;
    float* outputFrame;
    mallocFrames(&inputFrames, &outputFrame);

    vector<float> x = properties.get<vector<float>>("x");
    vector<float> y{ properties.get<vector<float>>("y_channel1"), properties.get<vector<float>>("y_channel2") };

    size_t frameToProcessCount = x.size() / FrameSampleCount;
    size_t currentFrameIndex = 0;
    for (size_t i = 0; i < frameToProcessCount; i++)
    {
        float* currentInputFrame = inputFrames + currentFrameIndex * ChannelCount * FrameSampleCount;
        memcpy(currentInputFrame, x.data() + i * FrameSampleCount, FrameSampleCount);
        memcpy(currentInputFrame + FrameSampleCount, x.data() + i * FrameSampleCount, FrameSampleCount);

        processEqKernel<<<1, 64>>>(eqBuffers, inputFrames, outputFrame, currentFrameIndex);
        cudaDeviceSynchronize();

        for (size_t channelIndex = 0; channelIndex < ChannelCount; channelIndex++)
        {
            for (size_t sampleIndex = 0; sampleIndex < FrameSampleCount; sampleIndex++)
            {
                float expectedValue = y[channelIndex][i * FrameSampleCount + sampleIndex];
                float value = outputFrame[channelIndex * FrameSampleCount + sampleIndex];
                ASSERT_NEAR(expectedValue, value, expectedValue * MaxAbsErrorFactor);
            }
        }

        currentFrameIndex = (currentFrameIndex + 1) % FrameCount;
    }
}
