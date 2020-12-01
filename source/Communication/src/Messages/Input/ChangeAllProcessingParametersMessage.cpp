#include <Communication/Messages/Input/ChangeAllProcessingParametersMessage.h>

#include <utility>

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
    vector<double> eqGains) :
    m_channelId(channelId),
    m_gain(gain),
    m_isMuted(isMuted),
    m_isSolo(isSolo),
    m_eqGains(move(eqGains))
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
    vector<ChannelGain> inputs,
    vector<double> eqGains) :
    m_gain(gain),
    m_isMuted(isMuted),
    m_inputs(move(inputs)),
    m_eqGains(move(eqGains))
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
    vector<ChannelGain> inputs,
    vector<double> eqGains) :
    m_auxiliaryChannelId(auxiliaryChannelId),
    m_gain(gain),
    m_isMuted(isMuted),
    m_inputs(move(inputs)),
    m_eqGains(move(eqGains))
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
    vector<InputProcessingParameters> inputs,
    MasterProcessingParameters master,
    vector<AuxiliaryProcessingParameters> auxiliaries,
    vector<size_t> inputChannelIds,
    size_t speakersNumber,
    vector<size_t> auxiliaryChannelIds) : ApplicationMessage(SeqId),
    m_inputs(move(inputs)),
    m_master(master),
    m_auxiliaries(move(auxiliaries)),
    m_inputChannelIds(move(inputChannelIds)),
    m_speakersNumber(speakersNumber),
    m_auxiliaryChannelIds(move(auxiliaryChannelIds))
{

}

ChangeAllProcessingParametersMessage::~ChangeAllProcessingParametersMessage()
{
}
