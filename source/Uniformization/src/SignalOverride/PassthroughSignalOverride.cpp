#include <Uniformization/SignalOverride/PassthroughSignalOverride.h>

using namespace adaptone;

PassthroughSignalOverride::PassthroughSignalOverride()
{
}

PassthroughSignalOverride::~PassthroughSignalOverride()
{
}

const PcmAudioFrame& PassthroughSignalOverride::override(const PcmAudioFrame& frame)
{
    return frame;
}
