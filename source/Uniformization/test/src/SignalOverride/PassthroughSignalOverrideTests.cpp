#include <Uniformization/SignalOverride/PassthroughSignalOverride.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(PassthroughSignalOverrideTests, override_shouldNotCopyData)
{
    PassthroughSignalOverride signalOverride;
    PcmAudioFrame frame(PcmAudioFrameFormat::Unsigned8, 2, 3);

    EXPECT_EQ(&frame, &signalOverride.override(frame));
}
