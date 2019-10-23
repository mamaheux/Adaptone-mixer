#ifndef MIXER_CONFIGURATION_CONFIGURATION_H
#define MIXER_CONFIGURATION_CONFIGURATION_H

#include <Mixer/Configuration/LoggerConfiguration.h>
#include <Mixer/Configuration/AudioConfiguration.h>
#include <Mixer/Configuration/AudioInputConfiguration.h>
#include <Mixer/Configuration/AudioOutputConfiguration.h>
#include <Mixer/Configuration/UniformizationConfiguration.h>
#include <Mixer/Configuration/WebSocketConfiguration.h>

#include <SignalProcessing/SignalProcessorParameters.h>
#include <Uniformization/UniformizationServiceParameters.h>

#include <Utils/Configuration/Properties.h>

namespace adaptone
{
    class Configuration
    {
        LoggerConfiguration m_loggerConfiguration;

        AudioConfiguration m_audioConfiguration;
        AudioInputConfiguration m_audioInputConfiguration;
        AudioOutputConfiguration m_audioOutputConfiguration;
        UniformizationConfiguration m_uniformizationConfiguration;
        WebSocketConfiguration m_webSocketConfiguration;

    public:
        explicit Configuration(const Properties& properties);
        virtual ~Configuration();

        const LoggerConfiguration& logger() const;
        const AudioConfiguration& audio() const;
        const AudioInputConfiguration& audioInput() const;
        const AudioOutputConfiguration& audioOutput() const;
        const UniformizationConfiguration& uniformization() const;
        const WebSocketConfiguration webSocket() const;

        SignalProcessorParameters toSignalProcessorParameters() const;
        UniformizationServiceParameters toUniformizationServiceParameters() const;
    };

    inline const LoggerConfiguration& Configuration::logger() const
    {
        return m_loggerConfiguration;
    }

    inline const AudioConfiguration& Configuration::audio() const
    {
        return m_audioConfiguration;
    }

    inline const AudioInputConfiguration& Configuration::audioInput() const
    {
        return m_audioInputConfiguration;
    }

    inline const AudioOutputConfiguration& Configuration::audioOutput() const
    {
        return m_audioOutputConfiguration;
    }

    inline const UniformizationConfiguration& Configuration::uniformization() const
    {
        return m_uniformizationConfiguration;
    }

    inline const WebSocketConfiguration Configuration::webSocket() const
    {
        return m_webSocketConfiguration;
    }

    inline SignalProcessorParameters Configuration::toSignalProcessorParameters() const
    {
        return SignalProcessorParameters(m_audioConfiguration.processingDataType(),
            m_audioConfiguration.frameSampleCount(),
            m_audioConfiguration.sampleFrequency(),
            m_audioConfiguration.inputChannelCount(),
            m_audioConfiguration.outputChannelCount(),
            m_audioInputConfiguration.format(),
            m_audioOutputConfiguration.format(),
            m_audioConfiguration.eqCenterFrequencies(),
            m_audioConfiguration.maxOutputDelay(),
            m_audioConfiguration.soundLevelLength());
    }

    inline UniformizationServiceParameters Configuration::toUniformizationServiceParameters() const
    {
        return UniformizationServiceParameters(m_uniformizationConfiguration.discoveryEndpoint(),
            m_uniformizationConfiguration.discoveryTimeoutMs(),
            m_uniformizationConfiguration.discoveryTrialCount(),
            m_uniformizationConfiguration.tcpConnectionPort(),
            m_uniformizationConfiguration.udpReceivingPort(),
            m_uniformizationConfiguration.probeTimeoutMs(),
            m_audioConfiguration.sampleFrequency(),
            m_uniformizationConfiguration.routineIRSweepT(),
            m_uniformizationConfiguration.routineIRSweepMaxDelay(),
            m_audioOutputConfiguration.hardwareDelay(),
            m_uniformizationConfiguration.speedOfSound(),
            m_audioOutputConfiguration.format());
    }
}

#endif
