#include <Mixer/MixerAnalysisDispatcher.h>

#include <Utils/Exception/NotSupportedException.h>

using namespace adaptone;
using namespace std;

constexpr size_t MixerAnalysisDispatcher::SoundLevelBoundedBufferSize;
constexpr size_t MixerAnalysisDispatcher::InputSampleBoundedBufferSize;

MixerAnalysisDispatcher::MixerAnalysisDispatcher(shared_ptr<Logger> logger,
    function<void(const ApplicationMessage&)> send,
    ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount) :
    m_logger(logger),
    m_send(send),
    m_processingDataType(processingDataType),
    m_soundLevelBoundedBuffer(SoundLevelBoundedBufferSize),
    m_floatInputSampleBoundedBuffer(InputSampleBoundedBufferSize,
        [=]() { return new float[frameSampleCount * inputChannelCount]; }, [](float*& b) { delete[] b; }),
    m_doubleInputSampleBoundedBuffer(InputSampleBoundedBufferSize,
        [=]() { return new double[frameSampleCount * inputChannelCount]; }, [](double*& b) { delete[] b; })
{
}

MixerAnalysisDispatcher::~MixerAnalysisDispatcher()
{
    if (!m_stopped.load())
    {
        stop();
    }
}

void MixerAnalysisDispatcher::start()
{
    m_stopped.store(false);
    startSoundLevelThread();
    startInputSampleThread();
}

void MixerAnalysisDispatcher::stop()
{
    bool wasStopped = m_stopped.load();
    m_stopped.store(true);
    if (!wasStopped)
    {
        stopSoundLevelThread();
        stopInputSampleThread();
    }
}

void MixerAnalysisDispatcher::notifySoundLevel(const map<SoundLevelType, vector<float>>& soundLevels)
{
    const vector<float>& inputAfterGain = soundLevels.at(SoundLevelType::InputGain);
    const vector<float>& inputAfterEq = soundLevels.at(SoundLevelType::InputEq);
    const vector<float>& outputAfterGain = soundLevels.at(SoundLevelType::OutputGain);

    m_soundLevelBoundedBuffer.write([&](SoundLevelMessage& m)
    {
        m = SoundLevelMessage(vector<double>(inputAfterGain.cbegin(), inputAfterGain.cend()),
            vector<double>(inputAfterEq.cbegin(), inputAfterEq.cend()),
            vector<double>(outputAfterGain.cbegin(), outputAfterGain.cend()));
    });
}

void MixerAnalysisDispatcher::notifySoundLevel(const map<SoundLevelType, vector<double>>& soundLevels)
{
    m_soundLevelBoundedBuffer.write([&](SoundLevelMessage& m)
    {
        m = SoundLevelMessage(soundLevels.at(SoundLevelType::InputGain),
            soundLevels.at(SoundLevelType::InputEq),
            soundLevels.at(SoundLevelType::OutputGain));
    });
}

void MixerAnalysisDispatcher::notifyInputEqOutputFrame(const function<void(float*)> notifyFunction)
{
    if (m_processingDataType != ProcessingDataType::Float)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Invalid processing data type");
    }

    m_floatInputSampleBoundedBuffer.write([&](float*& buffer)
    {
        notifyFunction(buffer);
    });
}

void MixerAnalysisDispatcher::notifyInputEqOutputFrame(const function<void(double*)> notifyFunction)
{
    if (m_processingDataType != ProcessingDataType::Double)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Invalid processing data type");
    }

    m_doubleInputSampleBoundedBuffer.write([&](double*& buffer)
    {
        notifyFunction(buffer);
    });
}

void MixerAnalysisDispatcher::soundLevelRun()
{
    while (!m_stopped.load())
    {
        try
        {
            m_soundLevelBoundedBuffer.read([&](const SoundLevelMessage& m)
            {
                if (!m_stopped.load())
                {
                    m_send(m);
                }
            });
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void MixerAnalysisDispatcher::floatInputSampleRun()
{
    while (!m_stopped.load())
    {
        try
        {
            //TODO add write call
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void MixerAnalysisDispatcher::doubleInputSampleRun()
{
    while (!m_stopped.load())
    {
        try
        {
            //TODO add write call
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void MixerAnalysisDispatcher::startSoundLevelThread()
{
    m_soundLevelThread = make_unique<thread>(&MixerAnalysisDispatcher::soundLevelRun, this);
}

void MixerAnalysisDispatcher::startInputSampleThread()
{
    switch (m_processingDataType)
    {
        case ProcessingDataType::Float:
            m_inputSampleThread = make_unique<thread>(&MixerAnalysisDispatcher::floatInputSampleRun, this);
            break;

        case ProcessingDataType::Double:
            m_inputSampleThread = make_unique<thread>(&MixerAnalysisDispatcher::doubleInputSampleRun, this);
            break;

        default:
            THROW_NOT_SUPPORTED_EXCEPTION("Invalid processing data type");
    }
}

void MixerAnalysisDispatcher::stopSoundLevelThread()
{
    // Write an empty message to unlock the sound level thread.
    m_soundLevelBoundedBuffer.write([](SoundLevelMessage& m) { });
    m_soundLevelThread->join();
    m_soundLevelThread.release();
}

void MixerAnalysisDispatcher::stopInputSampleThread()
{
    // Write an empty message to unlock the input sample thread.
    switch (m_processingDataType)
    {
        case ProcessingDataType::Float:
            m_floatInputSampleBoundedBuffer.write([](float*& m) { });
            break;

        case ProcessingDataType::Double:
            m_doubleInputSampleBoundedBuffer.write([](double*& m) { });
            break;

        default:
            THROW_NOT_SUPPORTED_EXCEPTION("Invalid processing data type");
    }

    m_inputSampleThread->join();
    m_inputSampleThread.release();
}
