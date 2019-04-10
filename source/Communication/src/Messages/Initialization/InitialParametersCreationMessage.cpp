#include <Communication/Messages/Initialization/InitialParametersCreationMessage.h>

using namespace adaptone;
using namespace std;

InitialParametersCreationMessage::InitialParametersCreationMessage() : ApplicationMessage(1),
    m_id(0),
    m_name(""),
    m_monitorsNumber(0),
    m_speakersNumber(0),
    m_probesNumber(0)
{
}

InitialParametersCreationMessage::InitialParametersCreationMessage(size_t id,
    const string& name,
    size_t monitorsNumber,
    size_t speakersNumber,
    size_t probesNumber) : ApplicationMessage(1),
    m_id(id),
    m_name(name),
    m_monitorsNumber(monitorsNumber),
    m_speakersNumber(speakersNumber),
    m_probesNumber(probesNumber)
{
}

InitialParametersCreationMessage::~InitialParametersCreationMessage()
{
}
