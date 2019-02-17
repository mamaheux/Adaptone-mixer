#ifndef SIGNAL_PROCESSING_SPECIFIC_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_SPECIFIC_SIGNAL_PROCESSOR_H

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class SpecificSignalProcessor
    {
    public:
        SpecificSignalProcessor();
        virtual ~SpecificSignalProcessor();

        DECLARE_NOT_COPYABLE(SpecificSignalProcessor);
        DECLARE_NOT_MOVABLE(SpecificSignalProcessor);

        virtual const PcmAudioFrame& process(const PcmAudioFrame& inputFrame);
    };
}

#endif
