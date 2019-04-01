#ifndef SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>
#include <SignalProcessing/SpecificSignalProcessor.h>
#include <SignalProcessing/Filters/ParametricEqParameters.h>

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
            std::size_t eqParametricFilterCount,
            const std::vector<double>& eqCenterFrequencies);
        virtual ~SignalProcessor();

        DECLARE_NOT_COPYABLE(SignalProcessor);
        DECLARE_NOT_MOVABLE(SignalProcessor);

        void setInputGain(std::size_t channel, double gainDb);
        void setInputGains(const std::vector<double>& gainsDb);

        void setInputParametricEqParameters(std::size_t channel, const std::vector<ParametricEqParameters>& parameters);
        void setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb);

        void setMixingGain(std::size_t inputChannel, std::size_t outputChannel, double gainDb);
        void setMixingGains(std::size_t outputChannel, const std::vector<double>& gainsDb);
        void setMixingGains(const std::vector<double>& gainsDb);

        void setOutputParametricEqParameters(std::size_t channel,
            const std::vector<ParametricEqParameters>& parameters);
        void setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb);

        void setOutputGain(std::size_t channel, double gainDb);
        void setOutputGains(const std::vector<double>& gainsDb);

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };

    inline void SignalProcessor::setInputGain(std::size_t channel, double gainDb)
    {
        m_specificSignalProcessor->setInputGain(channel, gainDb);
    }

    inline void SignalProcessor::setInputGains(const std::vector<double>& gainsDb)
    {
        m_specificSignalProcessor->setInputGains(gainsDb);
    }

    inline void SignalProcessor::setInputParametricEqParameters(std::size_t channel,
        const std::vector<ParametricEqParameters>& parameters)
    {
        m_specificSignalProcessor->setInputParametricEqParameters(channel, parameters);
    }

    inline void SignalProcessor::setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
    {
        m_specificSignalProcessor->setInputGraphicEqGains(channel, gainsDb);
    }

    inline void SignalProcessor::setMixingGain(std::size_t inputChannel, std::size_t outputChannel, double gainDb)
    {
        m_specificSignalProcessor->setMixingGain(inputChannel, outputChannel, gainDb);
    }

    inline void SignalProcessor::setMixingGains(std::size_t outputChannel, const std::vector<double>& gainsDb)
    {
        m_specificSignalProcessor->setMixingGains(outputChannel, gainsDb);
    }

    inline void SignalProcessor::setMixingGains(const std::vector<double>& gainsDb)
    {
        m_specificSignalProcessor->setMixingGains(gainsDb);
    }

    inline void SignalProcessor::setOutputParametricEqParameters(std::size_t channel,
        const std::vector<ParametricEqParameters>& parameters)
    {
        m_specificSignalProcessor->setOutputParametricEqParameters(channel, parameters);
    }

    inline void SignalProcessor::setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
    {
        m_specificSignalProcessor->setOutputGraphicEqGains(channel, gainsDb);
    }

    inline void SignalProcessor::setOutputGain(std::size_t channel, double gainDb)
    {
        m_specificSignalProcessor->setOutputGain(channel, gainDb);
    }

    inline void SignalProcessor::setOutputGains(const std::vector<double>& gainsDb)
    {
        m_specificSignalProcessor->setOutputGains(gainsDb);
    }

    inline const PcmAudioFrame& SignalProcessor::process(const PcmAudioFrame& inputFrame)
    {
        return m_specificSignalProcessor->process(inputFrame);
    }
}

#endif
