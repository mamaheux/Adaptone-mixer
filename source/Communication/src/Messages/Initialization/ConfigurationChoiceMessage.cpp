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
    string name,
    vector<size_t> inputChannelIds,
    size_t speakersNumber,
    vector<size_t> auxiliaryChannelIds,
    vector<ConfigurationPosition> positions) : ApplicationMessage(SeqId),
    m_id(id),
    m_name(move(name)),
    m_inputChannelIds(move(inputChannelIds)),
    m_speakersNumber(speakersNumber),
    m_auxiliaryChannelIds(move(auxiliaryChannelIds)),
    m_positions(move(positions))
{
}

ConfigurationChoiceMessage::~ConfigurationChoiceMessage()
{
}
