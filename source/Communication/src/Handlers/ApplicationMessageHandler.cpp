#include <Communication/Handlers/ApplicationMessageHandler.h>

#include <Communication/Messages/Initialization/ConfigurationChoiceMessage.h>
#include <Communication/Messages/Initialization/InitialParametersCreationMessage.h>
#include <Communication/Messages/Initialization/LaunchInitializationMessage.h>
#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>
#include <Communication/Messages/Initialization/RelaunchInitializationMessage.h>
#include <Communication/Messages/Initialization/SymmetryConfirmationMessage.h>
#include <Communication/Messages/Initialization/OptimizePositionMessage.h>
#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>
#include <Communication/Messages/Initialization/ReoptimizePositionMessage.h>
#include <Communication/Messages/Initialization/ConfigurationConfirmationMessage.h>

#include <Communication/Messages/Input/ChangeInputGainMessage.h>
#include <Communication/Messages/Input/ChangeInputGainsMessage.h>
#include <Communication/Messages/Input/ChangeInputEqGainsMessage.h>
#include <Communication/Messages/Input/ChangeMasterMixInputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeMasterMixInputVolumesMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumesMessage.h>
#include <Communication/Messages/Input/ChangeMasterOutputEqGainsMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryOutputEqGainsMessage.h>
#include <Communication/Messages/Input/ChangeMasterOutputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryOutputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeAllProcessingParametersMessage.h>

#include <Communication/Messages/Output/SoundErrorMessage.h>
#include <Communication/Messages/Output/InputSpectrumMessage.h>
#include <Communication/Messages/Output/SoundLevelMessage.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

#define ADD_HANDLE_FUNCTION(type) m_handleFunctions[type::SeqId] = [this](const json& j, \
        const function<void(const ApplicationMessage&)>& send) \
    { \
        type message = j.get<type>(); \
        handleDeserialized(message, send);\
    }

ApplicationMessageHandler::ApplicationMessageHandler()
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

    ADD_HANDLE_FUNCTION(SoundErrorMessage);
    ADD_HANDLE_FUNCTION(InputSpectrumMessage);
    ADD_HANDLE_FUNCTION(SoundLevelMessage);
}

ApplicationMessageHandler::~ApplicationMessageHandler()
{
}

void ApplicationMessageHandler::handle(const json& j, const function<void(const ApplicationMessage&)>& send)
{
    size_t seqId = j.at("seqId").get<size_t>();
    m_handleFunctions.at(seqId)(j, send);
}
