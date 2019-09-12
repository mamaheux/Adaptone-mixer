#include <Communication/Messages/Output/SoundLevelMessage.h>

using namespace adaptone;
using namespace std;

ChannelSoundLevel::ChannelSoundLevel() : m_channelId(0), m_level(0)
{
}

ChannelSoundLevel::ChannelSoundLevel(size_t channelId, double level) : m_channelId(channelId), m_level(level)
{
}

ChannelSoundLevel::~ChannelSoundLevel()
{
}

constexpr size_t SoundLevelMessage::SeqId;

SoundLevelMessage::SoundLevelMessage() : ApplicationMessage(SeqId),
    m_inputAfterGain(),
    m_inputAfterEq(),
    m_outputAfterGain()
{
}

SoundLevelMessage::SoundLevelMessage(const vector<ChannelSoundLevel>& inputAfterGain,
    const vector<ChannelSoundLevel>& inputAfterEq,
    const vector<ChannelSoundLevel>& outputAfterGain) : ApplicationMessage(SeqId),
    m_inputAfterGain(inputAfterGain),
    m_inputAfterEq(inputAfterEq),
    m_outputAfterGain(outputAfterGain)
{
}

SoundLevelMessage::~SoundLevelMessage()
{
}
