#include <Mixer/MixerApplicationMessageHandler.h>

#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>
#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>

#include <Utils/Exception/NotSupportedException.h>

#include <thread>
#include <optional>

using namespace adaptone;
using namespace std;

#define ADD_HANDLE_FUNCTION(type) m_handleFunctions[type::SeqId] = [this](const ApplicationMessage& message, \
        const function<void(const ApplicationMessage&)>& send) \
    { \
        handle##type(dynamic_cast<const type&>(message), send);\
    }

MixerApplicationMessageHandler::MixerApplicationMessageHandler(shared_ptr<ChannelIdMapping> channelIdMapping,
    shared_ptr<SignalProcessor> signalProcessor,
    shared_ptr<UniformizationService> uniformizationService) :
    m_channelIdMapping(channelIdMapping),
    m_signalProcessor(signalProcessor),
    m_uniformizationService(uniformizationService)
{
    ADD_HANDLE_FUNCTION(ConfigurationChoiceMessage);
    ADD_HANDLE_FUNCTION(InitialParametersCreationMessage);
    ADD_HANDLE_FUNCTION(LaunchInitializationMessage);
    ADD_HANDLE_FUNCTION(RelaunchInitializationMessage);
    ADD_HANDLE_FUNCTION(SymmetryConfirmationMessage);
    ADD_HANDLE_FUNCTION(OptimizePositionMessage);
    ADD_HANDLE_FUNCTION(ReoptimizePositionMessage);
    ADD_HANDLE_FUNCTION(ConfigurationConfirmationMessage);

    ADD_HANDLE_FUNCTION(ChangeInputGainMessage);
    ADD_HANDLE_FUNCTION(ChangeInputGainsMessage);
    ADD_HANDLE_FUNCTION(ChangeInputEqGainsMessage);
    ADD_HANDLE_FUNCTION(ChangeMasterMixInputVolumeMessage);
    ADD_HANDLE_FUNCTION(ChangeMasterMixInputVolumesMessage);
    ADD_HANDLE_FUNCTION(ChangeAuxiliaryMixInputVolumeMessage);
    ADD_HANDLE_FUNCTION(ChangeAuxiliaryMixInputVolumesMessage);
    ADD_HANDLE_FUNCTION(ChangeMasterOutputEqGainsMessage);
    ADD_HANDLE_FUNCTION(ChangeAuxiliaryOutputEqGainsMessage);
    ADD_HANDLE_FUNCTION(ChangeMasterOutputVolumeMessage);
    ADD_HANDLE_FUNCTION(ChangeAuxiliaryOutputVolumeMessage);
    ADD_HANDLE_FUNCTION(ChangeAllProcessingParametersMessage);
}

MixerApplicationMessageHandler::~MixerApplicationMessageHandler()
{
}

void MixerApplicationMessageHandler::handleDeserialized(const ApplicationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    auto it = m_handleFunctions.find(message.seqId());
    if (it == m_handleFunctions.end())
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Not supported message");
    }
    it->second(message, send);
}

void MixerApplicationMessageHandler::handleConfigurationChoiceMessage(const ConfigurationChoiceMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_channelIdMapping->update(message.inputChannelIds(), message.auxiliaryChannelIds(), message.speakersNumber());
}

void MixerApplicationMessageHandler::handleInitialParametersCreationMessage(
    const InitialParametersCreationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_channelIdMapping->update(message.inputChannelIds(), message.auxiliaryChannelIds(), message.speakersNumber());
}

void MixerApplicationMessageHandler::handleLaunchInitializationMessage(const LaunchInitializationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    vector<size_t> masterOutputIndexes =  m_channelIdMapping->getMasterOutputIndexes();
    Room room = m_uniformizationService->initializeRoom(masterOutputIndexes);

    const arma::mat speakersPosMat = room.getSpeakersPosMat();
    const arma::mat probePosMat = room.getProbesPosMat();

    vector<ConfigurationPosition> firstSymmetryPositions;
    vector<ConfigurationPosition> secondSymmetryPositions;
    for (int i = 0; i < speakersPosMat.n_rows; i++)
    {
        firstSymmetryPositions.emplace_back(ConfigurationPosition(speakersPosMat(i,0), speakersPosMat(i,1),
            PositionType::Speaker));
        secondSymmetryPositions.emplace_back(ConfigurationPosition(-speakersPosMat(i,0), speakersPosMat(i,1),
            PositionType::Speaker));
    }

    for (int i = 0; i < probePosMat.n_rows; i++)
    {
        firstSymmetryPositions.emplace_back(ConfigurationPosition(probePosMat(i,0), probePosMat(i,1),
            PositionType::Probe));
        secondSymmetryPositions.emplace_back(ConfigurationPosition(-probePosMat(i,0), probePosMat(i,1),
            PositionType::Probe));
    }

    send(PositionConfirmationMessage(firstSymmetryPositions, secondSymmetryPositions));
}

void MixerApplicationMessageHandler::handleRelaunchInitializationMessage(const RelaunchInitializationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    handleLaunchInitializationMessage(LaunchInitializationMessage(), send);
}

void MixerApplicationMessageHandler::handleSymmetryConfirmationMessage(const SymmetryConfirmationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
}

void MixerApplicationMessageHandler::handleOptimizePositionMessage(const OptimizePositionMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(OptimizedPositionMessage({ ConfigurationPosition(-10, 10, PositionType::Speaker) }));
}

void MixerApplicationMessageHandler::handleReoptimizePositionMessage(const ReoptimizePositionMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(OptimizedPositionMessage({ ConfigurationPosition(0, 0, PositionType::Speaker),
        ConfigurationPosition(2.5, 0, PositionType::Speaker),
        ConfigurationPosition(5, 0, PositionType::Speaker),
        ConfigurationPosition(7.5, 0, PositionType::Speaker),
        ConfigurationPosition(10, 0, PositionType::Speaker),
        ConfigurationPosition(1.25, 10, PositionType::Probe),
        ConfigurationPosition(3.75, 10, PositionType::Probe),
        ConfigurationPosition(6.25, 10, PositionType::Probe),
        ConfigurationPosition(8.75, 10, PositionType::Probe) }));
}

void MixerApplicationMessageHandler::handleConfigurationConfirmationMessage(
    const ConfigurationConfirmationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
}

void MixerApplicationMessageHandler::handleChangeInputGainMessage(const ChangeInputGainMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    size_t channelIndex = m_channelIdMapping->getInputIndexFromChannelId(message.channelId());
    m_signalProcessor->setInputGain(channelIndex, message.gain());
}

void MixerApplicationMessageHandler::handleChangeInputGainsMessage(const ChangeInputGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    for (const ChannelGain& channelGain : message.gains())
    {
        size_t channelIndex = m_channelIdMapping->getInputIndexFromChannelId(channelGain.channelId());
        m_signalProcessor->setInputGain(channelIndex, channelGain.gain());
    }
}

void MixerApplicationMessageHandler::handleChangeInputEqGainsMessage(const ChangeInputEqGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    size_t channelIndex = m_channelIdMapping->getInputIndexFromChannelId(message.channelId());
    m_signalProcessor->setInputGraphicEqGains(channelIndex, message.gains());
}

void MixerApplicationMessageHandler::handleChangeMasterMixInputVolumeMessage(
    const ChangeMasterMixInputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    size_t inputChannelIndex = m_channelIdMapping->getInputIndexFromChannelId(message.channelId());
    for (size_t outputChannelIndex : m_channelIdMapping->getMasterOutputIndexes())
    {
        m_signalProcessor->setMixingGain(inputChannelIndex, outputChannelIndex, message.gain());
    }
}

void MixerApplicationMessageHandler::handleChangeMasterMixInputVolumesMessage(
    const ChangeMasterMixInputVolumesMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    for (size_t outputChannelIndex : m_channelIdMapping->getMasterOutputIndexes())
    {
        for (const ChannelGain& channelGain : message.gains())
        {
            size_t inputChannelIndex = m_channelIdMapping->getInputIndexFromChannelId(channelGain.channelId());
            m_signalProcessor->setMixingGain(inputChannelIndex, outputChannelIndex, channelGain.gain());
        }
    }
}

void MixerApplicationMessageHandler::handleChangeAuxiliaryMixInputVolumeMessage(
    const ChangeAuxiliaryMixInputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    size_t inputChannelIndex = m_channelIdMapping->getInputIndexFromChannelId(message.channelId());
    size_t outputChannelIndex = m_channelIdMapping->getAuxiliaryOutputIndexFromChannelId(message.auxiliaryChannelId());
    m_signalProcessor->setMixingGain(inputChannelIndex, outputChannelIndex, message.gain());
}

void MixerApplicationMessageHandler::handleChangeAuxiliaryMixInputVolumesMessage(
    const ChangeAuxiliaryMixInputVolumesMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    size_t outputChannelIndex = m_channelIdMapping->getAuxiliaryOutputIndexFromChannelId(message.auxiliaryChannelId());
    for (const ChannelGain& channelGain : message.gains())
    {
        size_t inputChannelIndex = m_channelIdMapping->getInputIndexFromChannelId(channelGain.channelId());
        m_signalProcessor->setMixingGain(inputChannelIndex, outputChannelIndex, channelGain.gain());
    }
}

void MixerApplicationMessageHandler::handleChangeMasterOutputEqGainsMessage(
    const ChangeMasterOutputEqGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    for (size_t outputChannelIndex : m_channelIdMapping->getMasterOutputIndexes())
    {
        m_signalProcessor->setOutputGraphicEqGains(outputChannelIndex, message.gains());
    }
}

void MixerApplicationMessageHandler::handleChangeAuxiliaryOutputEqGainsMessage(
    const ChangeAuxiliaryOutputEqGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    size_t outputChannelIndex = m_channelIdMapping->getAuxiliaryOutputIndexFromChannelId(message.channelId());
    m_signalProcessor->setOutputGraphicEqGains(outputChannelIndex, message.gains());
}

void MixerApplicationMessageHandler::handleChangeMasterOutputVolumeMessage(
    const ChangeMasterOutputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    for (size_t outputChannelIndex : m_channelIdMapping->getMasterOutputIndexes())
    {
        m_signalProcessor->setOutputGain(outputChannelIndex, message.gain());
    }
}

void MixerApplicationMessageHandler::handleChangeAuxiliaryOutputVolumeMessage(
    const ChangeAuxiliaryOutputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    size_t outputChannelIndex = m_channelIdMapping->getAuxiliaryOutputIndexFromChannelId(message.channelId());
    m_signalProcessor->setOutputGain(outputChannelIndex, message.gain());
}

void MixerApplicationMessageHandler::handleChangeAllProcessingParametersMessage(
    const ChangeAllProcessingParametersMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_channelIdMapping->update(message.inputChannelIds(), message.auxiliaryChannelIds(), message.speakersNumber());

    applyInputProcessingParameters(message.inputs());
    applyMasterProcessingParameters(message.master());

    for (const AuxiliaryProcessingParameters& auxiliary : message.auxiliaries())
    {
        applyAuxiliaryProcessingParameters(auxiliary);
    }
}

void MixerApplicationMessageHandler::applyInputProcessingParameters(const vector<InputProcessingParameters>& inputs)
{
    optional<size_t> soloChannelId;
    for (const InputProcessingParameters& input : inputs)
    {
        if (input.isSolo())
        {
            soloChannelId = input.channelId();
        }
    }

    for (const InputProcessingParameters& input : inputs)
    {
        size_t channelIndex = m_channelIdMapping->getInputIndexFromChannelId(input.channelId());

        if (input.isMuted() || (soloChannelId != nullopt && soloChannelId != input.channelId()))
        {
            m_signalProcessor->setInputGain(channelIndex, input.gain());
        }
        else
        {
            m_signalProcessor->setInputGain(channelIndex, input.gain());
        }

        m_signalProcessor->setInputGraphicEqGains(channelIndex, input.eqGains());
    }
}

void MixerApplicationMessageHandler::applyMasterProcessingParameters(const MasterProcessingParameters& master)
{
    for (size_t outputChannelIndex : m_channelIdMapping->getMasterOutputIndexes())
    {
        if (master.isMuted())
        {
            m_signalProcessor->setOutputGain(outputChannelIndex, 0);
        }
        else
        {
            m_signalProcessor->setOutputGain(outputChannelIndex, master.gain());
        }

        for (const ChannelGain& channelGain : master.inputs())
        {
            size_t inputChannelIndex = m_channelIdMapping->getInputIndexFromChannelId(channelGain.channelId());
            m_signalProcessor->setMixingGain(inputChannelIndex, outputChannelIndex, channelGain.gain());
        }

        m_signalProcessor->setOutputGraphicEqGains(outputChannelIndex, master.eqGains());
    }
}

void MixerApplicationMessageHandler::applyAuxiliaryProcessingParameters(const AuxiliaryProcessingParameters& auxiliary)
{
    size_t outputChannelIndex =
        m_channelIdMapping->getAuxiliaryOutputIndexFromChannelId(auxiliary.auxiliaryChannelId());

    if (auxiliary.isMuted())
    {
        m_signalProcessor->setOutputGain(outputChannelIndex, 0);
    }
    else
    {
        m_signalProcessor->setOutputGain(outputChannelIndex, auxiliary.gain());
    }

    for (const ChannelGain& channelGain : auxiliary.inputs())
    {
        size_t inputChannelIndex = m_channelIdMapping->getInputIndexFromChannelId(channelGain.channelId());
        m_signalProcessor->setMixingGain(inputChannelIndex, outputChannelIndex, channelGain.gain());
    }

    m_signalProcessor->setOutputGraphicEqGains(outputChannelIndex, auxiliary.eqGains());
}
