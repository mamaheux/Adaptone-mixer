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
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat,
            const std::vector<double>& eqCenterFrequencies,
            std::size_t soundLevelLength,
            std::shared_ptr<AnalysisDispatcher> analysisDispatcher);
        virtual ~SignalProcessor();

        DECLARE_NOT_COPYABLE(SignalProcessor);
        DECLARE_NOT_MOVABLE(SignalProcessor);

        void setInputGain(std::size_t channel, double gain);
        void setInputGains(const std::vector<double>& gains);

        void setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gains);

        void setMixingGain(std::size_t inputChannel, std::size_t outputChannel, double gain);
        void setMixingGains(std::size_t outputChannel, const std::vector<double>& gains);
        void setMixingGains(const std::vector<double>& gains);

        void setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gains);
        void setOutputGraphicEqGains(std::size_t startChannelIndex, std::size_t n, const std::vector<double>& gains);

        void setOutputGain(std::size_t channel, double gain);
        void setOutputGains(const std::vector<double>& gains);

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };

    inline void SignalProcessor::setInputGain(std::size_t channel, double gain)
    {
        m_specificSignalProcessor->setInputGain(channel, gain);
    }

    inline void SignalProcessor::setInputGains(const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setInputGains(gains);
    }

    inline void SignalProcessor::setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setInputGraphicEqGains(channel, gains);
    }

    inline void SignalProcessor::setMixingGain(std::size_t inputChannel, std::size_t outputChannel, double gain)
    {
        m_specificSignalProcessor->setMixingGain(inputChannel, outputChannel, gain);
    }

    inline void SignalProcessor::setMixingGains(std::size_t outputChannel, const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setMixingGains(outputChannel, gains);
    }

    inline void SignalProcessor::setMixingGains(const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setMixingGains(gains);
    }

    inline void SignalProcessor::setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setOutputGraphicEqGains(channel, gains);
    }

    inline void SignalProcessor::setOutputGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
        const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setOutputGraphicEqGains(startChannelIndex, n, gains);
    }

    inline void SignalProcessor::setOutputGain(std::size_t channel, double gain)
    {
        m_specificSignalProcessor->setOutputGain(channel, gain);
    }

    inline void SignalProcessor::setOutputGains(const std::vector<double>& gains)
    {
        m_specificSignalProcessor->setOutputGains(gains);
    }

    inline const PcmAudioFrame& SignalProcessor::process(const PcmAudioFrame& inputFrame)
    {
        return m_specificSignalProcessor->process(inputFrame);
    }
}

#endif
