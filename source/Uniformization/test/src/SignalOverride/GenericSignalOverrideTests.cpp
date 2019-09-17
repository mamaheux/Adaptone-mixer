#include <Uniformization/SignalOverride/GenericSignalOverride.h>
#include <Uniformization/SignalOverride/PassthroughSignalOverride.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

class DummySignalOverride : public SpecificSignalOverride
{
    PcmAudioFrame m_overridenFrame;
    uint8_t m_firstByte;
public:
    DummySignalOverride() : m_overridenFrame(PcmAudioFrame::Format::Unsigned8, 2, 3), m_firstByte(0)
    {
    }

    void setFirstByte(uint8_t firstByte)
    {
        m_firstByte = firstByte;
    }

    const PcmAudioFrame& override(const PcmAudioFrame& frame)
    {
        m_overridenFrame = frame;
        m_overridenFrame.clear();
        m_overridenFrame[0] = m_firstByte;
        return m_overridenFrame;
    }
};

TEST(GenericSignalOverrideTests, constructor_emptyVector_shouldThrowInvalidValueException)
{
    EXPECT_THROW(GenericSignalOverride({}), InvalidValueException);
}

TEST(GenericSignalOverrideTests, constructor_sameTypes_shouldThrowInvalidValueException)
{
    vector<unique_ptr<SpecificSignalOverride>> signalOverrides;
    signalOverrides.emplace_back(make_unique<DummySignalOverride>());
    signalOverrides.emplace_back(make_unique<DummySignalOverride>());

    EXPECT_THROW(GenericSignalOverride(move(signalOverrides)), InvalidValueException);
}

TEST(GenericSignalOverrideTests, setCurrentSignalOverrideType_invalidType_shouldThrowInvalidValueException)
{
    vector<unique_ptr<SpecificSignalOverride>> signalOverrides;
    signalOverrides.emplace_back(make_unique<PassthroughSignalOverride>());
    signalOverrides.emplace_back(make_unique<DummySignalOverride>());

    GenericSignalOverride genericSignalOverride(move(signalOverrides));

    EXPECT_THROW(genericSignalOverride.setCurrentSignalOverrideType<vector<int>>(), InvalidValueException);
}

TEST(GenericSignalOverrideTests, getSignalOverride_invalidType_shouldThrowInvalidValueException)
{
    vector<unique_ptr<SpecificSignalOverride>> signalOverrides;
    signalOverrides.emplace_back(make_unique<PassthroughSignalOverride>());
    signalOverrides.emplace_back(make_unique<DummySignalOverride>());

    GenericSignalOverride genericSignalOverride(move(signalOverrides));

    EXPECT_THROW(genericSignalOverride.getSignalOverride<vector<int>>(), InvalidValueException);
}

TEST(GenericSignalOverrideTests, override_shouldCallTheCurrentSignalOverride)
{
    vector<unique_ptr<SpecificSignalOverride>> signalOverrides;
    signalOverrides.emplace_back(make_unique<PassthroughSignalOverride>());
    signalOverrides.emplace_back(make_unique<DummySignalOverride>());

    GenericSignalOverride genericSignalOverride(move(signalOverrides));

    PcmAudioFrame frame(PcmAudioFrame::Format::Unsigned8, 2, 3);
    for (size_t i = 0; i < frame.size(); i++)
    {
        frame[i] = i + 1;
    }

    const PcmAudioFrame& overridenFrame0 = genericSignalOverride.override(frame);
    for (size_t i = 0; i < frame.size(); i++)
    {
        EXPECT_EQ(frame[i], overridenFrame0[i]);
    }

    genericSignalOverride.setCurrentSignalOverrideType<DummySignalOverride>();
    const PcmAudioFrame& overridenFrame1 = genericSignalOverride.override(frame);
    for (size_t i = 0; i < frame.size(); i++)
    {
        EXPECT_EQ(overridenFrame1[i], 0);
    }

    genericSignalOverride.getSignalOverride<DummySignalOverride>().setFirstByte(1);
    const PcmAudioFrame& overridenFrame2 = genericSignalOverride.override(frame);
    EXPECT_EQ(overridenFrame2[0], 1);
    for (size_t i = 1; i < frame.size(); i++)
    {
        EXPECT_EQ(overridenFrame2[i], 0);
    }
}
