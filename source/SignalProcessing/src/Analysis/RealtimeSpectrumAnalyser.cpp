#include <SignalProcessing/Analysis/RealtimeSpectrumAnalyser.h>

#include <SignalProcessing/Analysis/Math.h>

#include <iostream>

using namespace adaptone;
using namespace std;

constexpr size_t RealtimeSpectrumAnalyser::BufferCount;

RealtimeSpectrumAnalyser::RealtimeSpectrumAnalyser(size_t fftSize,
    size_t sampleFrequency,
    size_t channelCount) :
    m_fftSize(fftSize),
    m_sampleFrequency(sampleFrequency),
    m_channelCount(channelCount),
    m_inputBufferSize(fftSize * channelCount),
    m_fftBufferSize((fftSize / 2 + 1) * channelCount),
    m_inputBoundedBuffers(BufferCount,
        [=]()
        { return fftwf_alloc_real(m_inputBufferSize); },
        [](float*& b)
        { fftwf_free(b); }),
    m_writingCount(0)
{
    constexpr int Rank = 1;
    const int N[] = { static_cast<int>(fftSize) };
    const int Howmany = static_cast<int>(channelCount);
    const int* Onembed = N;
    constexpr int Stride = 1;
    const int Dist = static_cast<int>(fftSize);

    for (float* inputBuffer : m_inputBoundedBuffers.buffers())
    {
        fftwf_complex* outputBuffer = fftwf_alloc_complex(m_fftBufferSize);
        m_fftBuffersByInputBuffer[inputBuffer] = outputBuffer;
        m_fftPlansByInputBuffer[inputBuffer] = fftwf_plan_many_dft_r2c(Rank, N, Howmany,
            inputBuffer, Onembed, Stride, Dist,
            outputBuffer, Onembed, Stride, Dist,
            FFTW_MEASURE);
        //m_fftPlansByInputBuffer[inputBuffer] = fftwf_plan_dft_r2c_1d(N[0], inputBuffer, outputBuffer, FFTW_MEASURE);
    }

    m_hammingWindows = arma::fvec(m_inputBufferSize);
    for (size_t i = 0; i < m_channelCount; i++)
    {
        m_hammingWindows(arma::span(i * fftSize, (i + 1) * fftSize - 1)) = hamming<arma::fvec>(fftSize);
    }
}

RealtimeSpectrumAnalyser::~RealtimeSpectrumAnalyser()
{
    for (const pair<float*, fftwf_plan>& keyValuePair : m_fftPlansByInputBuffer)
    {
        fftwf_destroy_plan(keyValuePair.second);
    }
    for (const pair<float*, fftwf_complex*>& keyValuePair : m_fftBuffersByInputBuffer)
    {
        fftwf_free(keyValuePair.second);
    }
}

arma::cx_fvec RealtimeSpectrumAnalyser::analyse()
{
    fftwf_complex* fft;
    m_inputBoundedBuffers.read([&](float* const& b)
    {
        float* rawBuffer = const_cast<float*&>(b);

        arma::fvec buffer(rawBuffer, m_inputBufferSize, false);
        buffer.print();
        cout << endl;

        //buffer %= m_hammingWindows;
        fftwf_execute(m_fftPlansByInputBuffer[rawBuffer]);
        fft = m_fftBuffersByInputBuffer[rawBuffer];
    });

    cout << "m_fftBufferSize=" << m_fftBufferSize << endl;
    cout << "m_inputBufferSize=" << m_inputBufferSize << endl;
    return arma::cx_fvec(reinterpret_cast<complex<float>*>(fft), m_fftBufferSize);
}

void RealtimeSpectrumAnalyser::writePartialData(function<void(size_t, float*)> writeFunction)
{
    m_inputBoundedBuffers.writePartialData([&](float*& buffer)
    {
        writeFunction(m_writingCount, buffer);
    });
    m_writingCount++;
}

void RealtimeSpectrumAnalyser::finishWriting()
{
    m_inputBoundedBuffers.finishWriting();
    m_writingCount = 0;
}
