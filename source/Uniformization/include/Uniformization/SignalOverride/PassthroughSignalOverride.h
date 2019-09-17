#ifndef UNIFORMIZATION_SIGNAL_OVERRIDE_PASSTHROUGH_SIGNAL_OVERRIDE_H
#define UNIFORMIZATION_SIGNAL_OVERRIDE_PASSTHROUGH_SIGNAL_OVERRIDE_H

#include <Uniformization/SignalOverride/SpecificSignalOverride.h>

namespace adaptone
{
    class PassthroughSignalOverride : public SpecificSignalOverride
    {
    public:
        PassthroughSignalOverride();
        ~PassthroughSignalOverride() override;

        DECLARE_NOT_COPYABLE(PassthroughSignalOverride);
        DECLARE_NOT_MOVABLE(PassthroughSignalOverride);

        const PcmAudioFrame& override(const PcmAudioFrame& frame) override;
    };
}

#endif
