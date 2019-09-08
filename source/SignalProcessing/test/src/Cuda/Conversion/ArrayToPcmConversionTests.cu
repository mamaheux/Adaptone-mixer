#include <SignalProcessing/Cuda/Conversion/ArrayToPcmConversion.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

template<class T>
__global__ void convertArrayToPcmKernel(const T* input, uint8_t* output, size_t frameSampleCount, size_t channelCount,
    PcmAudioFrame::Format format)
{
    convertArrayToPcm(input, output, frameSampleCount, channelCount, format);
}

TEST(ArrayToPcmConversionTests, convertSigned8_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 4;
    constexpr size_t ChannelCount = 2;
    double* input;
    int8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(double));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(int8_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = -1.1;

    input[4] = 0;
    input[5] = 0.5;
    input[6] = 0.25;
    input[7] = 1.1;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Signed8);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], -128);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 127);
    EXPECT_EQ(output[3], 64);
    EXPECT_EQ(output[4], -64);
    EXPECT_EQ(output[5], 32);
    EXPECT_EQ(output[6], -128);
    EXPECT_EQ(output[7], 127);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSigned16_shouldConvertTheDataFromFloatingPointArray)
{
    constexprsize_t FrameSampleCount = 4;
    constexprsize_t ChannelCount = 2;
    float* input;
    int16_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(int16_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = -1.1;

    input[4] = 0;
    input[5] = 0.5;
    input[6] = 0.25;
    input[7] = 1.1;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Signed16);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], -32768);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 32767);
    EXPECT_EQ(output[3], 16384);
    EXPECT_EQ(output[4], -16384);
    EXPECT_EQ(output[5], 8192);
    EXPECT_EQ(output[6], -32768);
    EXPECT_EQ(output[7], 32767);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSigned24_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    uint8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output),FframeSampleCount * ChannelCount * 3);

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, output, FrameSampleCount, ChannelCount, PcmAudioFrame::Format::Signed24);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], 0x00);
    EXPECT_EQ(output[1], 0x00);
    EXPECT_EQ(output[2], 0x80);

    EXPECT_EQ(output[3], 0x00);
    EXPECT_EQ(output[4], 0x00);
    EXPECT_EQ(output[5], 0x00);

    EXPECT_EQ(output[6], 0xff);
    EXPECT_EQ(output[7], 0xff);
    EXPECT_EQ(output[8], 0x7f);

    EXPECT_EQ(output[9], 0x00);
    EXPECT_EQ(output[10], 0x00);
    EXPECT_EQ(output[11], 0x40);

    EXPECT_EQ(output[12], 0x00);
    EXPECT_EQ(output[13], 0x00);
    EXPECT_EQ(output[14], 0xc0);

    EXPECT_EQ(output[15], 0x00);
    EXPECT_EQ(output[16], 0x00);
    EXPECT_EQ(output[17], 0x20);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSignedPadded24_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    int32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(int32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::SignedPadded24);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], -8388608);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 8388607);
    EXPECT_EQ(output[3], 4194304);
    EXPECT_EQ(output[4], -4194304);
    EXPECT_EQ(output[5], 2097152);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSigned32_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    int32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(int32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount,
        PcmAudioFrame::Format::Signed32);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], -2147483648);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 2147483647);
    EXPECT_EQ(output[3], 1073741824);
    EXPECT_EQ(output[4], -1073741824);
    EXPECT_EQ(output[5], 536870912);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned8_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    uint8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(uint8_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, output, FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Unsigned8);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 128);
    EXPECT_EQ(output[2], 255);
    EXPECT_EQ(output[3], 191);
    EXPECT_EQ(output[4], 64);
    EXPECT_EQ(output[5], 159);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned16_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    uint16_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(uint16_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Unsigned16);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 32768);
    EXPECT_EQ(output[2], 65535);
    EXPECT_EQ(output[3], 49151);
    EXPECT_EQ(output[4], 16384);
    EXPECT_EQ(output[5], 40959);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned24_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    uint8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * 3);


    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, output, FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Unsigned24);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], 0x00);
    EXPECT_EQ(output[1], 0x00);
    EXPECT_EQ(output[2], 0x00);

    EXPECT_EQ(output[3], 0x00);
    EXPECT_EQ(output[4], 0x00);
    EXPECT_EQ(output[5], 0x80);

    EXPECT_EQ(output[6], 0xff);
    EXPECT_EQ(output[7], 0xff);
    EXPECT_EQ(output[8], 0xff);

    EXPECT_EQ(output[9], 0xff);
    EXPECT_EQ(output[10], 0xff);
    EXPECT_EQ(output[11], 0xbf);

    EXPECT_EQ(output[12], 0x00);
    EXPECT_EQ(output[13], 0x00);
    EXPECT_EQ(output[14], 0x40);

    EXPECT_EQ(output[15], 0xff);
    EXPECT_EQ(output[16], 0xff);
    EXPECT_EQ(output[17], 0x9f);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsignedPadded24_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    uint32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(uint32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::UnsignedPadded24);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 8388608);
    EXPECT_EQ(output[2], 16777215);
    EXPECT_EQ(output[3], 12582911);
    EXPECT_EQ(output[4], 4194304);
    EXPECT_EQ(output[5], 10485759);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned32_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    uint32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(uint32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Unsigned32);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 2147483648);
    EXPECT_EQ(output[2], 4294967295);
    EXPECT_EQ(output[3], 3221225472);
    EXPECT_EQ(output[4], 1073741824);
    EXPECT_EQ(output[5], 2684354560);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertFloat_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 2;
    float* input;
    float* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(float));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Float);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], -1);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 1);
    EXPECT_EQ(output[3], 0.5);
    EXPECT_EQ(output[4], -0.5);
    EXPECT_EQ(output[5], 0.25);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertDouble_shouldConvertTheDataFromFloatingPointArray)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t FhannelCount = 2;
    float* input;
    double* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), FrameSampleCount * ChannelCount * sizeof(double));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convertArrayToPcmKernel<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), FrameSampleCount, ChannelCount,
        PcmAudioFrame::Format::Double);
    cudaDeviceSynchronize();

    EXPECT_EQ(output[0], -1);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 1);
    EXPECT_EQ(output[3], 0.5);
    EXPECT_EQ(output[4], -0.5);
    EXPECT_EQ(output[5], 0.25);

    cudaFree(input);
    cudaFree(output);
}
