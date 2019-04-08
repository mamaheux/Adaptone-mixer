#include <SignalProcessing/Cuda/Conversion/ArrayToPcmConversion.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

static constexpr double MaxAbsError = 0.01;

template<class T>
__global__ void convert(const T* inputBytes, uint8_t* output, std::size_t frameSampleCount, std::size_t channelCount,
    ArrayToPcmConversionFunctionPointer<T> conversionFunction)
{
    conversionFunction(inputBytes, output, frameSampleCount, channelCount);
}

TEST(ArrayToPcmConversionTests, convertSigned8_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Signed8);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    int8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(int8_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], -128, MaxAbsError);
    EXPECT_NEAR(output[1], 0, MaxAbsError);
    EXPECT_NEAR(output[2], 127, MaxAbsError);
    EXPECT_NEAR(output[3], 64, MaxAbsError);
    EXPECT_NEAR(output[4], -64, MaxAbsError);
    EXPECT_NEAR(output[5], 32, MaxAbsError);


    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSigned16_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Signed16);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    int16_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(int16_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], -32768, MaxAbsError);
    EXPECT_NEAR(output[1], 0, MaxAbsError);
    EXPECT_NEAR(output[2], 32767, MaxAbsError);
    EXPECT_NEAR(output[3], 16384, MaxAbsError);
    EXPECT_NEAR(output[4], -16384, MaxAbsError);
    EXPECT_NEAR(output[5], 8192, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSigned24_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Signed24);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    uint8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * 3);
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(uint8_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, output, frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], 0x00, MaxAbsError);
    EXPECT_NEAR(output[1], 0x00, MaxAbsError);
    EXPECT_NEAR(output[2], 0x80, MaxAbsError);

    EXPECT_NEAR(output[3], 0x00, MaxAbsError);
    EXPECT_NEAR(output[4], 0x00, MaxAbsError);
    EXPECT_NEAR(output[5], 0x00, MaxAbsError);

    EXPECT_NEAR(output[6], 0xff, MaxAbsError);
    EXPECT_NEAR(output[7], 0xff, MaxAbsError);
    EXPECT_NEAR(output[8], 0x7f, MaxAbsError);

    EXPECT_NEAR(output[9], 0x00, MaxAbsError);
    EXPECT_NEAR(output[10], 0x00, MaxAbsError);
    EXPECT_NEAR(output[11], 0x40, MaxAbsError);

    EXPECT_NEAR(output[12], 0x00, MaxAbsError);
    EXPECT_NEAR(output[13], 0x00, MaxAbsError);
    EXPECT_NEAR(output[14], 0xc0, MaxAbsError);

    EXPECT_NEAR(output[15], 0x00, MaxAbsError);
    EXPECT_NEAR(output[16], 0x00, MaxAbsError);
    EXPECT_NEAR(output[17], 0x20, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSignedPadded24_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::SignedPadded24);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    int32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(int32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], -8388608, MaxAbsError);
    EXPECT_NEAR(output[1], 0, MaxAbsError);
    EXPECT_NEAR(output[2], 8388607, MaxAbsError);
    EXPECT_NEAR(output[3], 4194304, MaxAbsError);
    EXPECT_NEAR(output[4], -4194304, MaxAbsError);
    EXPECT_NEAR(output[5], 2097152, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertSigned32_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Signed32);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    int32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(int32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], -2147483648, MaxAbsError);
    EXPECT_NEAR(output[1], 0, MaxAbsError);
    EXPECT_NEAR(output[2], 2147483647, MaxAbsError);

    EXPECT_NEAR(output[3], 1073741824, MaxAbsError);
    EXPECT_NEAR(output[4], -1073741824, MaxAbsError);
    EXPECT_NEAR(output[5], 536870912, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned8_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Unsigned8);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    uint8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(uint8_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, output, frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], 0, MaxAbsError);
    EXPECT_NEAR(output[1], 128, MaxAbsError);
    EXPECT_NEAR(output[2], 255, MaxAbsError);

    EXPECT_NEAR(output[3], 192, MaxAbsError);
    EXPECT_NEAR(output[4], 64, MaxAbsError);
    EXPECT_NEAR(output[5], 160, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned16_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Unsigned16);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    uint16_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(uint16_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], 0, MaxAbsError);
    EXPECT_NEAR(output[1], 32768, MaxAbsError);
    EXPECT_NEAR(output[2], 65535, MaxAbsError);

    EXPECT_NEAR(output[3], 49152, MaxAbsError);
    EXPECT_NEAR(output[4], 16384, MaxAbsError);
    EXPECT_NEAR(output[5], 40960, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned24_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Unsigned24);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    uint8_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * 3);
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(uint8_t));


    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;

    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, output, frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], 0x00, MaxAbsError);
    EXPECT_NEAR(output[1], 0x00, MaxAbsError);
    EXPECT_NEAR(output[2], 0x00, MaxAbsError);

    EXPECT_NEAR(output[3], 0x00, MaxAbsError);
    EXPECT_NEAR(output[4], 0x00, MaxAbsError);
    EXPECT_NEAR(output[5], 0x80, MaxAbsError);

    EXPECT_NEAR(output[6], 0xff, MaxAbsError);
    EXPECT_NEAR(output[7], 0xff, MaxAbsError);
    EXPECT_NEAR(output[8], 0xff, MaxAbsError);

    EXPECT_NEAR(output[9], 0x00, MaxAbsError);
    EXPECT_NEAR(output[10], 0x00, MaxAbsError);
    EXPECT_NEAR(output[11], 0xc0, MaxAbsError);

    EXPECT_NEAR(output[12], 0x00, MaxAbsError);
    EXPECT_NEAR(output[13], 0x00, MaxAbsError);
    EXPECT_NEAR(output[14], 0x40, MaxAbsError);

    EXPECT_NEAR(output[15], 0x00, MaxAbsError);
    EXPECT_NEAR(output[16], 0x00, MaxAbsError);
    EXPECT_NEAR(output[17], 0xa0, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsignedPadded24_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::UnsignedPadded24);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    uint32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(uint32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], 0, MaxAbsError);
    EXPECT_NEAR(output[1], 8388608, MaxAbsError);
    EXPECT_NEAR(output[2], 16777215, MaxAbsError);

    EXPECT_NEAR(output[3], 12582912, MaxAbsError);
    EXPECT_NEAR(output[4], 4194304, MaxAbsError);
    EXPECT_NEAR(output[5], 10485760, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertUnsigned32_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Unsigned32);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    uint32_t* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(uint32_t));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], 0, MaxAbsError);
    EXPECT_NEAR(output[1], 2147483648, MaxAbsError);
    EXPECT_NEAR(output[2], 4294967295, MaxAbsError);

    EXPECT_NEAR(output[3], 3221225472, MaxAbsError);
    EXPECT_NEAR(output[4], 1073741824, MaxAbsError);
    EXPECT_NEAR(output[5], 2684354560, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertFloat_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Float);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    float* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(float));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], -1, MaxAbsError);
    EXPECT_NEAR(output[1], 0, MaxAbsError);
    EXPECT_NEAR(output[2], 1, MaxAbsError);

    EXPECT_NEAR(output[3], 0.5, MaxAbsError);
    EXPECT_NEAR(output[4], -0.5, MaxAbsError);
    EXPECT_NEAR(output[5], 0.25, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}

TEST(ArrayToPcmConversionTests, convertDouble_shouldConvertTheDataFromFloatingPointArray)
{
    ArrayToPcmConversionFunctionPointer<float> conversionFunction =
        getArrayToPcmConversionFunctionPointer<float>(PcmAudioFrame::Format::Double);

    std::size_t frameSampleCount = 3;
    std::size_t channelCount = 2;
    float* input;
    double* output;

    cudaMallocManaged(reinterpret_cast<void**>(&input), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&output), frameSampleCount * channelCount * sizeof(double));

    input[0] = -1;
    input[1] = 1;
    input[2] = -0.5;
    input[3] = 0;
    input[4] = 0.5;
    input[5] = 0.25;

    convert<<<1, 256>>>(input, reinterpret_cast<uint8_t*>(output), frameSampleCount, channelCount, conversionFunction);
    cudaDeviceSynchronize();

    EXPECT_NEAR(output[0], -1, MaxAbsError);
    EXPECT_NEAR(output[1], 0, MaxAbsError);
    EXPECT_NEAR(output[2], 1, MaxAbsError);

    EXPECT_NEAR(output[3], 0.5, MaxAbsError);
    EXPECT_NEAR(output[4], -0.5, MaxAbsError);
    EXPECT_NEAR(output[5], 0.25, MaxAbsError);

    cudaFree(input);
    cudaFree(output);
}
