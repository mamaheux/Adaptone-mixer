#ifndef SIGNAL_PROCESSING_ANALYSIS_REALTIME_SPECTRUM_ANALYSER_H
#define SIGNAL_PROCESSING_ANALYSIS_REALTIME_SPECTRUM_ANALYSER_H

#include <Utils/Threading/OneWriterBoundedBuffer.h>

#include <armadillo>
#include <fftw3.h>

#include <map>

namespace adaptone
{
    class RealtimeSpectrumAnalyser
    {
        static constexpr std::size_t BufferCount = 2;

        std::size_t m_fftSize;
        std::size_t m_sampleFrequency;
        std::size_t m_channelCount;

        std::size_t m_inputBufferSize;
        std::size_t m_fftBufferSize;

        OneWriterBoundedBuffer<float*> m_inputBoundedBuffers;
        std::map<float*, fftwf_complex*> m_fftBuffersByInputBuffer;
        std::map<float*, fftwf_plan> m_fftPlansByInputBuffer;

        arma::fvec m_hammingWindows;
        std::size_t m_writingCount;

    public:
        RealtimeSpectrumAnalyser(std::size_t fftSize,
            std::size_t sampleFrequency,
            std::size_t channelCount);
        virtual ~RealtimeSpectrumAnalyser();

        arma::cx_fvec analyse();

        void writePartialData(std::function<void(std::size_t, float*)> writeFunction);
        void finishWriting();
    };
}

#endif
