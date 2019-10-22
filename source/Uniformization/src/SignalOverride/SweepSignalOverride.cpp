#include <Uniformization/SignalOverride/SweepSignalOverride.h>

using namespace adaptone;

SweepSignalOverride::SweepSignalOverride()
{
}

SweepSignalOverride::~SweepSignalOverride()
{
}

const PcmAudioFrame& SweepSignalOverride::override(const PcmAudioFrame& frame)
{
    return frame;
}
