#include <Communication/Messages/Initialization/InitialParametersCreationMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t InitialParametersCreationMessage::SeqId;

InitialParametersCreationMessage::InitialParametersCreationMessage() : ApplicationMessage(SeqId),
    m_id(0),
    m_name(""),
    m_inputChannelIds(),
    m_speakersNumber(0),
    m_auxiliaryChannelIds()
{
}

InitialParametersCreationMessage::InitialParametersCreationMessage(size_t id,
    string name,
    vector<size_t> inputChannelIds,
    size_t speakersNumber,
    vector<size_t> auxiliaryChannelIds) : ApplicationMessage(SeqId),
    m_id(id),
    m_name(move(name)),
    m_inputChannelIds(move(inputChannelIds)),
    m_speakersNumber(speakersNumber),
    m_auxiliaryChannelIds(move(auxiliaryChannelIds))
{
}

InitialParametersCreationMessage::~InitialParametersCreationMessage()
{
}
