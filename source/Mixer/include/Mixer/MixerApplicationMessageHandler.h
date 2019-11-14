#ifndef MIXER_MIXER_APPLICATION_MESSAGE_HANDLER_H
#define MIXER_MIXER_APPLICATION_MESSAGE_HANDLER_H

#include <Mixer/ChannelIdMapping.h>

#include <Communication/Handlers/ApplicationMessageHandler.h>

#include <Communication/Messages/Initialization/ConfigurationChoiceMessage.h>
#include <Communication/Messages/Initialization/InitialParametersCreationMessage.h>
#include <Communication/Messages/Initialization/LaunchInitializationMessage.h>
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
#include <Communication/Messages/Input/ListenProbeMessage.h>
#include <Communication/Messages/Input/StopProbeListeningMessage.h>

#include <SignalProcessing/SignalProcessor.h>

#include <Uniformization/UniformizationService.h>

#include <memory>

namespace adaptone
{
    class MixerApplicationMessageHandler : public ApplicationMessageHandler
    {
        std::shared_ptr<ChannelIdMapping> m_channelIdMapping;
        std::shared_ptr<SignalProcessor> m_signalProcessor;
        std::shared_ptr<UniformizationService> m_uniformizationService;

        std::size_t m_outputChannelCount;

        std::unordered_map<std::size_t,std::function<void(const ApplicationMessage&,
            const std::function<void(const ApplicationMessage&)>&)>> m_handleFunctions;

    public:
        MixerApplicationMessageHandler(std::shared_ptr<ChannelIdMapping> channelIdMapping,
            std::shared_ptr<SignalProcessor> signalProcessor,
            std::shared_ptr<UniformizationService> uniformizationService,
            size_t outputChannelCount);
        ~MixerApplicationMessageHandler() override;

        DECLARE_NOT_COPYABLE(MixerApplicationMessageHandler);
        DECLARE_NOT_MOVABLE(MixerApplicationMessageHandler);

        void handleDeserialized(const ApplicationMessage& message,
            const std::function<void(const ApplicationMessage&)>& send) override;

        void handleConfigurationChoiceMessage(const ConfigurationChoiceMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleInitialParametersCreationMessage(const InitialParametersCreationMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleLaunchInitializationMessage(const LaunchInitializationMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleRelaunchInitializationMessage(const RelaunchInitializationMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleSymmetryConfirmationMessage(const SymmetryConfirmationMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleOptimizePositionMessage(const OptimizePositionMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleReoptimizePositionMessage(const ReoptimizePositionMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleConfigurationConfirmationMessage(const ConfigurationConfirmationMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);

        void handleChangeInputGainMessage(const ChangeInputGainMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeInputGainsMessage(const ChangeInputGainsMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeInputEqGainsMessage(const ChangeInputEqGainsMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeMasterMixInputVolumeMessage(const ChangeMasterMixInputVolumeMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeMasterMixInputVolumesMessage(const ChangeMasterMixInputVolumesMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeAuxiliaryMixInputVolumeMessage(const ChangeAuxiliaryMixInputVolumeMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeAuxiliaryMixInputVolumesMessage(const ChangeAuxiliaryMixInputVolumesMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeMasterOutputEqGainsMessage(const ChangeMasterOutputEqGainsMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeAuxiliaryOutputEqGainsMessage(const ChangeAuxiliaryOutputEqGainsMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeMasterOutputVolumeMessage(const ChangeMasterOutputVolumeMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeAuxiliaryOutputVolumeMessage(const ChangeAuxiliaryOutputVolumeMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleChangeAllProcessingParametersMessage(const ChangeAllProcessingParametersMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleListenProbeMessage(const ListenProbeMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);
        void handleStopProbeListeningMessage(const StopProbeListeningMessage& message,
            const std::function<void(const ApplicationMessage&)>& send);

        void applyInputProcessingParameters(const std::vector<InputProcessingParameters>& inputs);
        void applyMasterProcessingParameters(const MasterProcessingParameters& master);
        void applyAuxiliaryProcessingParameters(const AuxiliaryProcessingParameters& auxiliary);
    };
}

#endif
