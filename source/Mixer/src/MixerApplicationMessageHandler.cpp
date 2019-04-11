#include <Mixer/MixerApplicationMessageHandler.h>

#include <thread>

using namespace adaptone;
using namespace std;

#define ADD_HANDLE_FUNCTION(type) m_handleFunctions[type::SeqId] = [this](const ApplicationMessage& message, \
        const function<void(const ApplicationMessage&)>& send) \
    { \
        handle##type(dynamic_cast<const type&>(message), send);\
    }

MixerApplicationMessageHandler::MixerApplicationMessageHandler(shared_ptr<SignalProcessor> signalProcessor) :
    m_signalProcessor(signalProcessor)
{
    ADD_HANDLE_FUNCTION(ConfigurationChoiceMessage);
    ADD_HANDLE_FUNCTION(InitialParametersCreationMessage);
    ADD_HANDLE_FUNCTION(LaunchInitializationMessage);
    ADD_HANDLE_FUNCTION(PositionConfirmationMessage);
    ADD_HANDLE_FUNCTION(RelaunchInitializationMessage);
    ADD_HANDLE_FUNCTION(SymmetryConfirmationMessage);
    ADD_HANDLE_FUNCTION(OptimizePositionMessage);
    ADD_HANDLE_FUNCTION(OptimizedPositionMessage);
    ADD_HANDLE_FUNCTION(ReoptimizePositionMessage);
    ADD_HANDLE_FUNCTION(ConfigurationConfirmationMessage);
}

MixerApplicationMessageHandler::~MixerApplicationMessageHandler()
{
}

void MixerApplicationMessageHandler::handleDeserialized(const ApplicationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    m_handleFunctions.at(message.seqId())(message, send);
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
    send(PositionConfirmationMessage({ ConfigurationPosition(-10, 10, ConfigurationPosition::Type::Speaker) },
        { ConfigurationPosition(10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handlePositionConfirmationMessage(const PositionConfirmationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
}

void MixerApplicationMessageHandler::handleRelaunchInitializationMessage(const RelaunchInitializationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(PositionConfirmationMessage({ ConfigurationPosition(-10, 10, ConfigurationPosition::Type::Speaker) },
        { ConfigurationPosition(10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleSymmetryConfirmationMessage(const SymmetryConfirmationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    send(PositionConfirmationMessage({ ConfigurationPosition(-10, 10, ConfigurationPosition::Type::Speaker) },
        { ConfigurationPosition(10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleOptimizePositionMessage(const OptimizePositionMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(OptimizedPositionMessage({ ConfigurationPosition(-10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleOptimizedPositionMessage(const OptimizedPositionMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
}

void MixerApplicationMessageHandler::handleReoptimizePositionMessage(const ReoptimizePositionMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
    this_thread::sleep_for(2s);
    send(OptimizedPositionMessage({ ConfigurationPosition(-10, 10, ConfigurationPosition::Type::Speaker) }));
}

void MixerApplicationMessageHandler::handleConfigurationConfirmationMessage(
    const ConfigurationConfirmationMessage& message,
    const function<void(const ApplicationMessage&)>& send)
{
}
