#ifndef SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>
#include <SignalProcessing/SpecificSignalProcessor.h>
#include <SignalProcessing/AnalysisDispatcher.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

#include <memory>

namespace adaptone
{
    class SignalProcessor
    {
        std::unique_ptr<SpecificSignalProcessor> m_specificSignalProcessor;

    public:
        SignalProcessor(ProcessingDataType processingDataType,
            std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrameFormat inputFormat,
            PcmAudioFrameFormat outputFormat,
            const std::vector<double>& eqCenterFrequencies,
            std::size_t maxOutputDelay,
            std::size_t soundLevelLength,
            std::shared_ptr<AnalysisDispatcher> analysisDispatcher);
        virtual ~SignalProcessor();

        DECLARE_NOT_COPYABLE(SignalProcessor);
        DECLARE_NOT_MOVABLE(SignalProcessor);

        void setInputGain(std::size_t channelIndex, double gain);
        void setInputGains(const std::vector<double>& gains);

        void setInputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains);

        void setMixingGain(std::size_t inputChannelIndex, std::size_t outputChannelIndex, double gain);
        void setMixingGains(std::size_t outputChannelIndex, const std::vector<double>& gains);
        void setMixingGains(const std::vector<double>& gains);

        void setOutputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains);
        void setOutputGraphicEqGains(std::size_t startChannelIndex, std::size_t n, const std::vector<double>& gains);

        void setOutputGain(std::size_t channelIndex, double gain);
        void setOutputGains(const std::vector<double>& gains);

        void setOutputDelay(std::size_t channelIndex, std::size_t delay);
        void setOutputDelays(const std::vector<std::size_t>& delays);

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };

    inline void SignalProcessor::setInputGain(std::size_t channelIndex, double gain)
    {
        m_specificSignalProcessor->setInputGain(channelIndex, gain);
    }

    inline void SignalProcessor::setInputGains(const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setInputGains(gains);
    }

    inline void SignalProcessor::setInputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setInputGraphicEqGains(channelIndex, gains);
    }

    inline void SignalProcessor::setMixingGain(std::size_t inputChannelIndex, std::size_t outputChannelIndex,
        double gain)
    {
        m_specificSignalProcessor->setMixingGain(inputChannelIndex, outputChannelIndex, gain);
    }

    inline void SignalProcessor::setMixingGains(std::size_t outputChannelIndex, const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setMixingGains(outputChannelIndex, gains);
    }

    inline void SignalProcessor::setMixingGains(const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setMixingGains(gains);
    }

    inline void SignalProcessor::setOutputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setOutputGraphicEqGains(channelIndex, gains);
    }

    inline void SignalProcessor::setOutputGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
        const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setOutputGraphicEqGains(startChannelIndex, n, gains);
    }

    inline void SignalProcessor::setOutputGain(std::size_t channelIndex, double gain)
    {
        m_specificSignalProcessor->setOutputGain(channelIndex, gain);
    }

    inline void SignalProcessor::setOutputGains(const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setOutputGains(gains);
    }

    inline void SignalProcessor::setOutputDelay(std::size_t channelIndex, std::size_t delay)
    {
        m_specificSignalProcessor->setOutputDelay(channelIndex, delay);
    }

    inline void SignalProcessor::setOutputDelays(const std::vector<std::size_t>& delays)
    {
        m_specificSignalProcessor->setOutputDelays(delays);
    }

    inline const PcmAudioFrame& SignalProcessor::process(const PcmAudioFrame& inputFrame)
    {
        return m_specificSignalProcessor->process(inputFrame);
    }
}

#endif
