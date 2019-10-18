#include <Communication/Messages/Input/ChangeAllProcessingParametersMessage.h>

using namespace adaptone;
using namespace std;

InputProcessingParameters::InputProcessingParameters() :
    m_channelId(0),
    m_gain(0),
    m_isMuted(false),
    m_isSolo(false)
{
}

InputProcessingParameters::InputProcessingParameters(size_t channelId,
    double gain,
    bool isMuted,
    bool isSolo,
    const vector<double>& eqGains) :
    m_channelId(channelId),
    m_gain(gain),
    m_isMuted(isMuted),
    m_isSolo(isSolo),
    m_eqGains(eqGains)
{
}

InputProcessingParameters::~InputProcessingParameters()
{
}

MasterProcessingParameters::MasterProcessingParameters() :
    m_gain(0),
    m_isMuted(false)
{
}

MasterProcessingParameters::MasterProcessingParameters(double gain,
    bool isMuted,
    const vector<ChannelGain>& inputs,
    const vector<double>& eqGains) :
    m_gain(gain),
    m_isMuted(isMuted),
    m_inputs(inputs),
    m_eqGains(eqGains)
{
}

MasterProcessingParameters::~MasterProcessingParameters()
{
}

AuxiliaryProcessingParameters::AuxiliaryProcessingParameters() :
    m_auxiliaryChannelId(0),
    m_gain(0),
    m_isMuted(0)
{
}

AuxiliaryProcessingParameters::AuxiliaryProcessingParameters(size_t auxiliaryChannelId,
    double gain,
    bool isMuted,
    const vector<ChannelGain>& inputs,
    const vector<double>& eqGains) :
    m_auxiliaryChannelId(auxiliaryChannelId),
    m_gain(gain),
    m_isMuted(isMuted),
    m_inputs(inputs),
    m_eqGains(eqGains)
{
}

AuxiliaryProcessingParameters::~AuxiliaryProcessingParameters()
{
}

constexpr size_t ChangeAllProcessingParametersMessage::SeqId;

ChangeAllProcessingParametersMessage::ChangeAllProcessingParametersMessage() :
    ApplicationMessage(SeqId),
    m_speakersNumber(0)
{
}

ChangeAllProcessingParametersMessage::ChangeAllProcessingParametersMessage(
    const vector<InputProcessingParameters>& inputs,
    const MasterProcessingParameters& master,
    const vector<AuxiliaryProcessingParameters>& auxiliaries,
    const vector<size_t>& inputChannelIds,
    size_t speakersNumber,
    const vector<size_t>& auxiliaryChannelIds) : ApplicationMessage(SeqId),
    m_inputs(inputs),
    m_master(master),
    m_auxiliaries(auxiliaries),
    m_inputChannelIds(inputChannelIds),
    m_speakersNumber(speakersNumber),
    m_auxiliaryChannelIds(auxiliaryChannelIds)
{

}

ChangeAllProcessingParametersMessage::~ChangeAllProcessingParametersMessage()
{
}
