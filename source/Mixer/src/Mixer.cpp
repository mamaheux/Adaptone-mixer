#include <Mixer/Mixer.h>

#include <Mixer/MixerAnalysisDispatcher.h>
#include <Mixer/MixerConnectionHandler.h>
#include <Mixer/MixerApplicationMessageHandler.h>
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

    shared_ptr<AnalysisDispatcher> analysisDispatcher = createAnalysisDispatcher();
    shared_ptr<SignalProcessor> signalProcessor = createSignalProcessor(analysisDispatcher);

    shared_ptr<ConnectionHandler> connectionHandler = createConnectionHandler(signalProcessor);
    shared_ptr<ApplicationMessageHandler> applicationMessageHandler = createApplicationMessageHandler(signalProcessor);
    unique_ptr<ApplicationWebSocket> applicationWebSocket = createApplicationWebSocket(logger,
        connectionHandler,
        applicationMessageHandler);

    //Create all members, then assign them to the attributes to prevent memory leaks
    m_logger = logger;

    m_audioInput = move(audioInput);
    m_audioOutput = move(audioOutput);

    m_signalProcessor = signalProcessor;
    m_analysisDispatcher = analysisDispatcher;

    m_connectionHandler = connectionHandler;
    m_applicationMessageHandler = applicationMessageHandler;
    m_applicationWebSocket = move(applicationWebSocket);
}

Mixer::~Mixer()
{
}

void Mixer::run()
{
    m_stopped.store(false);
    m_analysisThread = make_unique<thread>(&Mixer::analysisRun, this);
    m_applicationWebSocketThread = make_unique<thread>(&Mixer::applicationWebSocketRun, this);
    this_thread::sleep_for(1s); //Make sure the websocket is properly started.

    processingRun();

    m_analysisThread->join();
    m_applicationWebSocketThread->join();
}

void Mixer::stop()
{
    if (!m_stopped.load())
    {
        m_applicationWebSocket->stop();
    }

    m_stopped.store(true);
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

shared_ptr<AnalysisDispatcher> Mixer::createAnalysisDispatcher()
{
    return make_shared<MixerAnalysisDispatcher>();
}

shared_ptr<SignalProcessor> Mixer::createSignalProcessor(shared_ptr<AnalysisDispatcher> analysisDispatcher)
{
    return make_unique<SignalProcessor>(m_configuration.audio().processingDataType(),
        m_configuration.audio().frameSampleCount(),
        m_configuration.audio().sampleFrequency(),
        m_configuration.audio().inputChannelCount(),
        m_configuration.audio().outputChannelCount(),
        m_configuration.audioInput().format(),
        m_configuration.audioOutput().format(),
        m_configuration.audio().eqCenterFrequencies(),
        m_configuration.audio().soundLevelLength(),
        analysisDispatcher);
}

shared_ptr<ConnectionHandler> Mixer::createConnectionHandler(shared_ptr<SignalProcessor> signalProcessor)
{
    return make_shared<MixerConnectionHandler>(signalProcessor);
}

shared_ptr<ApplicationMessageHandler> Mixer::createApplicationMessageHandler(
    shared_ptr<SignalProcessor> signalProcessor)
{
    return make_shared<MixerApplicationMessageHandler>(signalProcessor);
}

unique_ptr<ApplicationWebSocket> Mixer::createApplicationWebSocket(shared_ptr<Logger> logger,
    shared_ptr<ConnectionHandler> connectionHandler,
    shared_ptr<ApplicationMessageHandler> applicationMessageHandler)
{
    return make_unique<ApplicationWebSocket>(logger,
        connectionHandler,
        applicationMessageHandler,
        m_configuration.webSocket().endpoint(),
        m_configuration.webSocket().port());
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
    stop();
}

void Mixer::analysisRun()
{
    try
    {
        while (!m_stopped.load())
        {
            this_thread::sleep_for(100ms);
            //TODO Add the analysis code
        }
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
    }
    stop();
}

void Mixer::applicationWebSocketRun()
{
    m_applicationWebSocket->start();
}
