#include <Mixer/MixerApplicationMessageHandler.h>

#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>
#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>

#include <Utils/Exception/NotSupportedException.h>

#include <thread>

using namespace adaptone;
using namespace std;

#define ADD_HANDLE_FUNCTION(type) m_handleFunctions[type::SeqId] = [this](const ApplicationMessage& message, \
        const function<void(const ApplicationMessage&)>& send) \
    { \
        handle##type(dynamic_cast<const type&>(message), send);\
    }

MixerApplicationMessageHandler::MixerApplicationMessageHandler(shared_ptr<SignalProcessor> signalProcessor,
    size_t outputChannelCount) :
    m_signalProcessor(signalProcessor),
    m_masterOutputCount(2), //TODO Obtenir la valeur à partir de la configuration de l'application
    m_outputChannelCount(outputChannelCount)
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
    ADD_HANDLE_FUNCTION(ChangeAuxiliaryMixInputVolumeMessage);
    ADD_HANDLE_FUNCTION(ChangeMasterOutputEqGainsMessage);
    ADD_HANDLE_FUNCTION(ChangeAuxiliaryOutputEqGainsMessage);
    ADD_HANDLE_FUNCTION(ChangeMasterOutputVolumeMessage);
    ADD_HANDLE_FUNCTION(ChangeAuxiliaryOutputVolumeMessage);
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
}

void MixerApplicationMessageHandler::handleInitialParametersCreationMessage(
    const InitialParametersCreationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
}

void MixerApplicationMessageHandler::handleLaunchInitializationMessage(const LaunchInitializationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(PositionConfirmationMessage({ ConfigurationPosition(0, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(2.5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(7.5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(10, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(1.25, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(3.75, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(6.25, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(8.75, 10, ConfigurationPosition::Type::Probe) },
        { ConfigurationPosition(10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleRelaunchInitializationMessage(const RelaunchInitializationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(PositionConfirmationMessage({ ConfigurationPosition(0, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(2.5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(7.5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(10, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(1.25, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(3.75, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(6.25, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(8.75, 10, ConfigurationPosition::Type::Probe) },
        { ConfigurationPosition(10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleSymmetryConfirmationMessage(const SymmetryConfirmationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    send(PositionConfirmationMessage({ ConfigurationPosition(0, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(2.5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(7.5, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(10, 0, ConfigurationPosition::Type::Speaker),
            ConfigurationPosition(1.25, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(3.75, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(6.25, 10, ConfigurationPosition::Type::Probe),
            ConfigurationPosition(8.75, 10, ConfigurationPosition::Type::Probe) },
        { ConfigurationPosition(10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleOptimizePositionMessage(const OptimizePositionMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(OptimizedPositionMessage({ ConfigurationPosition(-10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleReoptimizePositionMessage(const ReoptimizePositionMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(OptimizedPositionMessage({ ConfigurationPosition(0, 0, ConfigurationPosition::Type::Speaker),
        ConfigurationPosition(2.5, 0, ConfigurationPosition::Type::Speaker),
        ConfigurationPosition(5, 0, ConfigurationPosition::Type::Speaker),
        ConfigurationPosition(7.5, 0, ConfigurationPosition::Type::Speaker),
        ConfigurationPosition(10, 0, ConfigurationPosition::Type::Speaker),
        ConfigurationPosition(1.25, 10, ConfigurationPosition::Type::Probe),
        ConfigurationPosition(3.75, 10, ConfigurationPosition::Type::Probe),
        ConfigurationPosition(6.25, 10, ConfigurationPosition::Type::Probe),
        ConfigurationPosition(8.75, 10, ConfigurationPosition::Type::Probe) }));
}

void MixerApplicationMessageHandler::handleConfigurationConfirmationMessage(
    const ConfigurationConfirmationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
}

void MixerApplicationMessageHandler::handleChangeInputGainMessage(const ChangeInputGainMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_signalProcessor->setInputGain(message.channelId(), message.gain());
}

void MixerApplicationMessageHandler::handleChangeInputGainsMessage(const ChangeInputGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_signalProcessor->setInputGains(message.gains());
}

void MixerApplicationMessageHandler::handleChangeInputEqGainsMessage(const ChangeInputEqGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_signalProcessor->setInputGraphicEqGains(message.channelId(), message.gains());
}

void MixerApplicationMessageHandler::handleChangeMasterMixInputVolumeMessage(
    const ChangeMasterMixInputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    for (size_t outputChannel = 0; outputChannel < m_masterOutputCount; outputChannel++)
    {
        m_signalProcessor->setMixingGain(message.channelId(), outputChannel, message.gain());
    }
}

void MixerApplicationMessageHandler::handleChangeAuxiliaryMixInputVolumeMessage(
    const ChangeAuxiliaryMixInputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_signalProcessor->setMixingGain(message.channelId(),
        getOutputChannelFromAuxiliaryId(message.auxiliaryId()),
        message.gain());
}

void MixerApplicationMessageHandler::handleChangeMasterOutputEqGainsMessage(
    const ChangeMasterOutputEqGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_signalProcessor->setOutputGraphicEqGains(0, m_masterOutputCount, message.gains());
}

void MixerApplicationMessageHandler::handleChangeAuxiliaryOutputEqGainsMessage(
    const ChangeAuxiliaryOutputEqGainsMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_signalProcessor->setOutputGraphicEqGains(getOutputChannelFromAuxiliaryId(message.auxiliaryId()),
        message.gains());
}

void MixerApplicationMessageHandler::handleChangeMasterOutputVolumeMessage(
    const ChangeMasterOutputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    for (size_t outputChannel = 0; outputChannel < m_masterOutputCount; outputChannel++)
    {
        m_signalProcessor->setOutputGain(outputChannel, message.gain());
    }
}

void MixerApplicationMessageHandler::handleChangeAuxiliaryOutputVolumeMessage(
    const ChangeAuxiliaryOutputVolumeMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_signalProcessor->setOutputGain(getOutputChannelFromAuxiliaryId(message.auxiliaryId()), message.gain());
}

size_t MixerApplicationMessageHandler::getOutputChannelFromAuxiliaryId(size_t auxiliaryId)
{
    return m_outputChannelCount - auxiliaryId - 1;
}
