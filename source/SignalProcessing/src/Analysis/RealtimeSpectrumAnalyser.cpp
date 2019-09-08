#include <SignalProcessing/Analysis/RealtimeSpectrumAnalyser.h>

#include <SignalProcessing/Analysis/Math.h>

using namespace adaptone;
using namespace std;

constexpr size_t RealtimeSpectrumAnalyser::BufferCount;

RealtimeSpectrumAnalyser::RealtimeSpectrumAnalyser(size_t inputFftSize,
    size_t sampleFrequency,
    size_t channelCount,
    size_t decimatorPointCountPerDecade) :
    m_inputFftSize(inputFftSize),
    m_outputFftSize(inputFftSize / 2 + 1),
    m_sampleFrequency(sampleFrequency),
    m_channelCount(channelCount),
    m_inputBufferSize(inputFftSize * channelCount),
    m_fftBufferSize((inputFftSize / 2 + 1) * channelCount),
    m_inputBoundedBuffers(BufferCount,
        [=]()
        { return fftwf_alloc_real(m_inputBufferSize); },
        [](float*& b)
        { fftwf_free(b); }),
    m_writingCount(0),
    m_spectrumDecimator(m_outputFftSize, sampleFrequency, decimatorPointCountPerDecade)
{
    constexpr int Rank = 1;
    const int N[] = { static_cast<int>(inputFftSize) };
    const int Howmany = static_cast<int>(channelCount);
    const int* Onembed = N;
    constexpr int Stride = 1;
    const int InDist = static_cast<int>(inputFftSize);
    const int OutDist = static_cast<int>(m_outputFftSize);

    for (float* inputBuffer : m_inputBoundedBuffers.buffers())
    {
        fftwf_complex* outputBuffer = fftwf_alloc_complex(m_fftBufferSize);
        m_fftBuffersByInputBuffer[inputBuffer] = outputBuffer;
        m_fftPlansByInputBuffer[inputBuffer] = fftwf_plan_many_dft_r2c(Rank, N, Howmany,
            inputBuffer, Onembed, Stride, InDist,
            outputBuffer, Onembed, Stride, OutDist,
            FFTW_PATIENT);
    }

    m_hammingWindows = arma::fvec(m_inputBufferSize);
    for (size_t i = 0; i < m_channelCount; i++)
    {
        m_hammingWindows(arma::span(i * inputFftSize, (i + 1) * inputFftSize - 1)) = hamming<arma::fvec>(inputFftSize);
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

vector<arma::cx_fvec> RealtimeSpectrumAnalyser::calculateFftAnalysis()
{
    complex<float>* fft = reinterpret_cast<complex<float>*>(analyse());
    vector<arma::cx_fvec> channelFfts;
    channelFfts.reserve(m_channelCount);

    for (size_t i = 0; i < m_channelCount; i++)
    {
        channelFfts.push_back(arma::cx_fvec(fft + i * m_outputFftSize, m_outputFftSize));
    }

    return channelFfts;
}

std::vector<std::vector<SpectrumPoint>> RealtimeSpectrumAnalyser::calculateDecimatedSpectrumAnalysis()
{
    complex<float>* fft = reinterpret_cast<complex<float>*>(analyse());
    std::vector<std::vector<SpectrumPoint>> decimatedSpectrumAnalysisResult;
    decimatedSpectrumAnalysisResult.reserve(m_channelCount);

    for (size_t i = 0; i < m_channelCount; i++)
    {
        arma::fvec amplitudes = arma::abs(arma::cx_fvec(fft + i * m_outputFftSize, m_outputFftSize));
        decimatedSpectrumAnalysisResult.push_back(m_spectrumDecimator.getDecimatedAmplitudes(amplitudes));
    }

    return decimatedSpectrumAnalysisResult;
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

fftwf_complex* RealtimeSpectrumAnalyser::analyse()
{
    fftwf_complex* fft;
    m_inputBoundedBuffers.read([&](float* const& b)
    {
        float* rawBuffer = const_cast<float*&>(b);

        arma::fvec buffer(rawBuffer, m_inputBufferSize, false);

        buffer %= m_hammingWindows;
        fftwf_execute(m_fftPlansByInputBuffer[rawBuffer]);
        fft = m_fftBuffersByInputBuffer[rawBuffer];
    });

    return fft;
}
