#ifndef UNIFORMIZATION_SIGNAL_OVERRIDE_SPECIFIC_SIGNAL_OVERRIDE_H
#define UNIFORMIZATION_SIGNAL_OVERRIDE_SPECIFIC_SIGNAL_OVERRIDE_H

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class SpecificSignalOverride
    {
    public:
        SpecificSignalOverride();
        virtual ~SpecificSignalOverride();

        DECLARE_NOT_COPYABLE(SpecificSignalOverride);
        DECLARE_NOT_MOVABLE(SpecificSignalOverride);

        virtual const PcmAudioFrame& override(const PcmAudioFrame& frame) = 0;
    };
}

#endif
