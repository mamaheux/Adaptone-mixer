#ifndef SIGNAL_PROCESSING_SPECIFIC_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_SPECIFIC_SIGNAL_PROCESSOR_H

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

#include <vector>

namespace adaptone
{
    class SpecificSignalProcessor
    {
    public:
        SpecificSignalProcessor();
        virtual ~SpecificSignalProcessor();

        DECLARE_NOT_COPYABLE(SpecificSignalProcessor);
        DECLARE_NOT_MOVABLE(SpecificSignalProcessor);

        virtual void setInputGain(std::size_t channel, double gain);
        virtual void setInputGains(const std::vector<double>& gains);

        virtual void setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gains);

        virtual void setMixingGain(std::size_t inputChannel, std::size_t outputChannel, double gain);
        virtual void setMixingGains(std::size_t outputChannel, const std::vector<double>& gains);
        virtual void setMixingGains(const std::vector<double>& gains);

        virtual void setOutputGain(std::size_t channel, double gain);
        virtual void setOutputGains(const std::vector<double>& gains);

        virtual void setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gains);
        virtual void setOutputGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
            const std::vector<double>& gains);

        virtual const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };
}

#endif
