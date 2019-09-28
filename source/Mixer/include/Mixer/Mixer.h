#ifndef MIXER_MIXER_H
#define MIXER_MIXER_H

#include <Mixer/Configuration/Configuration.h>
#include <Mixer/AudioInput/AudioInput.h>
#include <Mixer/AudioOutput/AudioOutput.h>
#include <Mixer/ChannelIdMapping.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <SignalProcessing/SignalProcessor.h>
#include <SignalProcessing/AnalysisDispatcher.h>

#include <Uniformization/SignalOverride/GenericSignalOverride.h>
#include <Uniformization/UniformizationService.h>

#include <Communication/ApplicationWebSocket.h>
#include <Communication/Handlers/ConnectionHandler.h>
#include <Communication/Handlers/ApplicationMessageHandler.h>

#include <memory>
#include <atomic>
#include <thread>

namespace adaptone
{
    class Mixer
    {
        Configuration m_configuration;
        std::shared_ptr<Logger> m_logger;
        std::shared_ptr<ChannelIdMapping> m_channelIdMapping;

        std::unique_ptr<AudioInput> m_audioInput;
        std::unique_ptr<AudioOutput> m_audioOutput;

        std::shared_ptr<AnalysisDispatcher> m_analysisDispatcher;
        std::shared_ptr<SignalProcessor> m_signalProcessor;
        std::shared_ptr<GenericSignalOverride> m_outputSignalOverride;

        std::unique_ptr<UniformizationService> m_uniformizationService;

        std::shared_ptr<ConnectionHandler> m_connectionHandler;
        std::shared_ptr<ApplicationMessageHandler> m_applicationMessageHandler;
        std::shared_ptr<ApplicationWebSocket> m_applicationWebSocket;

        std::unique_ptr<std::thread> m_applicationWebSocketThread;
        std::atomic<bool> m_stopped;

    public:
        Mixer(const Configuration& configuration);
        virtual ~Mixer();

        DECLARE_NOT_COPYABLE(Mixer);
        DECLARE_NOT_MOVABLE(Mixer);

        void run();
        void stop();

    private:
        std::shared_ptr<Logger> createLogger();
        std::shared_ptr<ChannelIdMapping> createChannelIdMapping();

        std::unique_ptr<AudioInput> createAudioInput();
        std::unique_ptr<AudioOutput> createAudioOutput();

        std::shared_ptr<AnalysisDispatcher> createAnalysisDispatcher(std::shared_ptr<Logger> logger,
            std::shared_ptr<ChannelIdMapping> channelIdMapping);
        std::shared_ptr<SignalProcessor> createSignalProcessor(std::shared_ptr<AnalysisDispatcher> analysisDispatcher);
        std::shared_ptr<GenericSignalOverride> createOutputSignalOverride();

        std::unique_ptr<UniformizationService> createUniformizationService(std::shared_ptr<Logger> logger,
            std::shared_ptr<GenericSignalOverride> outputSignalOverride);

        std::shared_ptr<ConnectionHandler> createConnectionHandler(std::shared_ptr<SignalProcessor> signalProcessor);
        std::shared_ptr<ApplicationMessageHandler> createApplicationMessageHandler(
            std::shared_ptr<ChannelIdMapping> channelIdMapping,
            std::shared_ptr<SignalProcessor> signalProcessor);
        std::unique_ptr<ApplicationWebSocket> createApplicationWebSocket(std::shared_ptr<Logger> logger,
            std::shared_ptr<ConnectionHandler> connectionHandler,
            std::shared_ptr<ApplicationMessageHandler> applicationMessageHandler);

        void processingRun();
        void applicationWebSocketRun();
    };
}

#endif
