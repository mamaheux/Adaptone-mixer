#include <Communication/Messages/Output/SoundLevelMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t SoundLevelMessage::SeqId;

SoundLevelMessage::SoundLevelMessage() : ApplicationMessage(SeqId),
    m_inputAfterGain(),
    m_inputAfterEq(),
    m_outputAfterGain()
{
}

SoundLevelMessage::SoundLevelMessage(const vector<double>& inputAfterGain,
    const vector<double>& inputAfterEq,
    const vector<double>& outputAfterGain) : ApplicationMessage(SeqId),
    m_inputAfterGain(inputAfterGain),
    m_inputAfterEq(inputAfterEq),
    m_outputAfterGain(outputAfterGain)
{
}

SoundLevelMessage::~SoundLevelMessage()
{
}
