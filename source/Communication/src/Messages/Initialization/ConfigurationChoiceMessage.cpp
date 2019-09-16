#include <Communication/Messages/Initialization/ConfigurationChoiceMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ConfigurationChoiceMessage::SeqId;

ConfigurationChoiceMessage::ConfigurationChoiceMessage() : ApplicationMessage(SeqId),
    m_id(0),
    m_name(""),
    m_inputChannelIds(),
    m_speakersNumber(0),
    m_auxiliaryChannelIds(),
    m_positions()
{
}

ConfigurationChoiceMessage::ConfigurationChoiceMessage(size_t id,
    const string& name,
    const vector<size_t>& inputChannelIds,
    size_t speakersNumber,
    const vector<size_t>& auxiliaryChannelIds,
    const vector<ConfigurationPosition>& positions) : ApplicationMessage(SeqId),
    m_id(id),
    m_name(name),
    m_inputChannelIds(inputChannelIds),
    m_speakersNumber(speakersNumber),
    m_auxiliaryChannelIds(auxiliaryChannelIds),
    m_positions(positions)
{
}

ConfigurationChoiceMessage::~ConfigurationChoiceMessage()
{
}
