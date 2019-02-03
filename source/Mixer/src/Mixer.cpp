#include <Mixer/Mixer.h>

#include <Mixer/AudioInput/RawFileAudioInput.h>

#include <Utils/Exception/NotSupportedException.h>
#include <Utils/Logger/ConsoleLogger.h>
#include <Utils/Logger/FileLogger.h>

using namespace adaptone;
using namespace std;

Mixer::Mixer(const Configuration& configuration) : m_configuration(configuration)
{
    shared_ptr<Logger> logger = createLogger();
    unique_ptr<AudioInput> audioInput = createAudioInput();


    //Create all members, then assign them to the attributes to prevent memory leaks
    m_logger = logger;
    m_audioInput = move(audioInput);
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
    }

    THROW_NOT_SUPPORTED_EXCEPTION("Not supported audio input type.");
}