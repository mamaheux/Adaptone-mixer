#include <Mixer/Mixer.h>

#include <Mixer/AudioInput/RawFileAudioInput.h>
#include <Mixer/AudioOutput/RawFileAudioOutput.h>

#if defined(__unix__) || defined(__linux__)

#include <Mixer/AudioInput/AlsaAudioInput.h>
#include <Mixer/AudioOutput/AlsaAudioOutput.h>

#endif

#include <Utils/Exception/NotSupportedException.h>
#include <Utils/Logger/ConsoleLogger.h>
#include <Utils/Logger/FileLogger.h>

using namespace adaptone;
using namespace std;

Mixer::Mixer(const Configuration& configuration) : m_configuration(configuration)
{
    shared_ptr<Logger> logger = createLogger();

    unique_ptr<AudioInput> audioInput = createAudioInput();
    unique_ptr<AudioOutput> audioOutput = createAudioOutput();


    //Create all members, then assign them to the attributes to prevent memory leaks
    m_logger = logger;

    m_audioInput = move(audioInput);
    m_audioOutput = move(audioOutput);
}

Mixer::~Mixer()
{
}

int Mixer::run()
{
    try
    {
        return 0;
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
        return -1;
    }
}

std::shared_ptr<Logger> Mixer::createLogger()
{
    switch (m_configuration.logger().type())
    {
        case LoggerConfiguration::Type::Console:
            return make_shared<ConsoleLogger>();

        case LoggerConfiguration::Type::File:
            return make_shared<FileLogger>(m_configuration.logger().filename());
    }

    THROW_NOT_SUPPORTED_EXCEPTION("Not supported logger type.");
}

std::unique_ptr<AudioInput> Mixer::createAudioInput()
{
    switch (m_configuration.audioInput().type())
    {
        case AudioInputConfiguration::Type::RawFile:
            return make_unique<RawFileAudioInput>(m_configuration.audioInput().format(),
                m_configuration.audio().inputChannelCount(),
                m_configuration.audio().frameSampleCount(),
                m_configuration.audioInput().filename(),
                m_configuration.audioInput().looping());

#if defined(__unix__) || defined(__linux__)
        case AudioInputConfiguration::Type::Alsa:
            return make_unique<AlsaAudioInput>(m_configuration.audioInput().format(),
                m_configuration.audio().inputChannelCount(),
                m_configuration.audio().frameSampleCount(),
                m_configuration.audio().sampleFrequency(),
                m_configuration.audioInput().device());
#endif
    }

    THROW_NOT_SUPPORTED_EXCEPTION("Not supported audio input type.");
}

std::unique_ptr<AudioOutput> Mixer::createAudioOutput()
{
    switch (m_configuration.audioOutput().type())
    {
        case AudioOutputConfiguration::Type::RawFile:
            return make_unique<RawFileAudioOutput>(m_configuration.audioOutput().format(),
                m_configuration.audio().inputChannelCount(),
                m_configuration.audio().frameSampleCount(),
                m_configuration.audioOutput().filename());

#if defined(__unix__) || defined(__linux__)
        case AudioOutputConfiguration::Type::Alsa:
            return make_unique<AlsaAudioOutput>(m_configuration.audioOutput().format(),
                m_configuration.audio().inputChannelCount(),
                m_configuration.audio().frameSampleCount(),
                m_configuration.audio().sampleFrequency(),
                m_configuration.audioOutput().device());
#endif
    }

    THROW_NOT_SUPPORTED_EXCEPTION("Not supported audio input type.");
}
