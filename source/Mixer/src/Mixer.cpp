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

Mixer::Mixer(const Configuration& configuration) : m_configuration(configuration), m_stopped(false)
{
    shared_ptr<Logger> logger = createLogger();

    unique_ptr<AudioInput> audioInput = createAudioInput();
    unique_ptr<AudioOutput> audioOutput = createAudioOutput();

    unique_ptr<SignalProcessor> signalProcessor = createSignalProcessor();

    //Create all members, then assign them to the attributes to prevent memory leaks
    m_logger = logger;

    m_audioInput = move(audioInput);
    m_audioOutput = move(audioOutput);

    m_signalProcessor = move(signalProcessor);
}

Mixer::~Mixer()
{
}

void Mixer::run()
{
    m_stopped.store(false);
    m_analysisThread = make_unique<thread>(&Mixer::analysisRun, this);
    //TODO Add the websocket start

    processingRun();

    m_analysisThread->join();
}

void Mixer::stop()
{
    m_stopped.store(true);
    //TODO Add the websocket stop
}

shared_ptr<Logger> Mixer::createLogger()
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

unique_ptr<AudioInput> Mixer::createAudioInput()
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

unique_ptr<AudioOutput> Mixer::createAudioOutput()
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

unique_ptr<SignalProcessor> Mixer::createSignalProcessor()
{
    return make_unique<SignalProcessor>(m_configuration.audio().processingDataType(),
        m_configuration.audio().frameSampleCount(),
        m_configuration.audio().sampleFrequency(),
        m_configuration.audio().inputChannelCount(),
        m_configuration.audio().outputChannelCount(),
        m_configuration.audioInput().format(),
        m_configuration.audioOutput().format());
}

void Mixer::analysisRun()
{
    try
    {
        while (!m_stopped.load())
        {
            //TODO Add the analysis code
        }
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
    }
    m_stopped.store(true);
}

void Mixer::processingRun()
{
    try
    {
        while (!m_stopped.load() && m_audioInput->hasNext())
        {
            const PcmAudioFrame& inputFrame = m_audioInput->read();
            const PcmAudioFrame& outputFrame = m_signalProcessor->process(inputFrame);
            m_audioOutput->write(outputFrame);
        }
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
    }
    m_stopped.store(true);
}
