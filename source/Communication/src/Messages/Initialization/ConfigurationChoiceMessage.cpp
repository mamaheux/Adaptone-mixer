#include <Communication/Messages/Initialization/ConfigurationChoiceMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ConfigurationChoiceMessage::SeqId;

ConfigurationChoiceMessage::ConfigurationChoiceMessage() : ApplicationMessage(SeqId),
    m_id(0),
    m_name(""),
    m_monitorsNumber(0),
    m_speakersNumber(0),
    m_probesNumber(0),
    m_positions()
{
}

ConfigurationChoiceMessage::ConfigurationChoiceMessage(size_t id,
    const string& name,
    size_t monitorsNumber,
    size_t speakersNumber,
    size_t probesNumber,
    const vector<ConfigurationPosition>& positions) : ApplicationMessage(SeqId),
    m_id(id),
    m_name(name),
    m_monitorsNumber(monitorsNumber),
    m_speakersNumber(speakersNumber),
    m_probesNumber(probesNumber),
    m_positions(positions)
{
}

ConfigurationChoiceMessage::~ConfigurationChoiceMessage()
{
}
