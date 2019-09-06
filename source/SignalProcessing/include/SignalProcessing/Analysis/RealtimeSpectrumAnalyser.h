#ifndef SIGNAL_PROCESSING_ANALYSIS_REALTIME_SPECTRUM_ANALYSER_H
#define SIGNAL_PROCESSING_ANALYSIS_REALTIME_SPECTRUM_ANALYSER_H

#include <SignalProcessing/Analysis/SpectrumDecimator.h>

#include <Utils/Threading/OneWriterBoundedBuffer.h>

#include <armadillo>
#include <fftw3.h>

#include <map>
#include <vector>

namespace adaptone
{
    class RealtimeSpectrumAnalyser
    {
        static constexpr std::size_t BufferCount = 2;

        std::size_t m_inputFftSize;
        std::size_t m_outputFftSize;
        std::size_t m_sampleFrequency;
        std::size_t m_channelCount;

        std::size_t m_inputBufferSize;
        std::size_t m_fftBufferSize;

        OneWriterBoundedBuffer<float*> m_inputBoundedBuffers;
        std::map<float*, fftwf_complex*> m_fftBuffersByInputBuffer;
        std::map<float*, fftwf_plan> m_fftPlansByInputBuffer;

        arma::fvec m_hammingWindows;
        std::size_t m_writingCount;

        SpectrumDecimator m_spectrumDecimator;

    public:
        RealtimeSpectrumAnalyser(std::size_t inputFftSize,
            std::size_t sampleFrequency,
            std::size_t channelCount,
            std::size_t decimatorPointCountPerDecade);
        virtual ~RealtimeSpectrumAnalyser();

        std::vector<arma::cx_fvec> calculateFftAnalysis();
        std::vector<std::vector<SpectrumPoint>> calculateDecimatedSpectrumAnalysis();

        void writePartialData(std::function<void(std::size_t, float*)> writeFunction);
        void finishWriting();

    private:
        fftwf_complex* analyse();
    };
}

#endif
