#ifndef UNIFORMIZATION_MODEL_SIGNAL_OVERRIDE_SWEEP_SIGNAL_OVERRIDE_H
#define UNIFORMIZATION_MODEL_SIGNAL_OVERRIDE_SWEEP_SIGNAL_OVERRIDE_H

#include <Uniformization/SignalOverride/SpecificSignalOverride.h>

namespace adaptone
{
    class SweepSignalOverride : public SpecificSignalOverride
    {
    public:
        SweepSignalOverride();
        ~SweepSignalOverride() override;

        DECLARE_NOT_COPYABLE(SweepSignalOverride);
        DECLARE_NOT_MOVABLE(SweepSignalOverride);

        const PcmAudioFrame& override(const PcmAudioFrame& frame) override;
    };
}

#endif
