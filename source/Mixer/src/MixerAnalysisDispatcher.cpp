#include <Mixer/MixerAnalysisDispatcher.h>

#include <Communication/Messages/Output/InputSpectrumMessage.h>

#include <Utils/Exception/NotSupportedException.h>

#include <cstring>

using namespace adaptone;
using namespace std;

constexpr size_t MixerAnalysisDispatcher::SoundLevelBoundedBufferSize;
constexpr size_t MixerAnalysisDispatcher::InputSampleBoundedBufferSize;

MixerAnalysisDispatcher::MixerAnalysisDispatcher(shared_ptr<Logger> logger,
    function<void(const ApplicationMessage&)> send,
    ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount,
    size_t spectrumAnalysisFftLength,
    size_t spectrumAnalysisPointCountPerDecade) :
    m_logger(logger),
    m_send(send),
    m_processingDataType(processingDataType),
    m_frameSampleCount(frameSampleCount),
    m_inputChannelCount(inputChannelCount),
    m_spectrumAnalysisFftLength(spectrumAnalysisFftLength),
    m_soundLevelBoundedBuffer(SoundLevelBoundedBufferSize),
    m_floatInputEqOutputFrameBoundedBuffer(InputSampleBoundedBufferSize,
        [=]() { return new float[frameSampleCount * inputChannelCount]; }, [](float*& b) { delete[] b; }),
    m_doubleInputEqOutputFrameBoundedBuffer(InputSampleBoundedBufferSize,
        [=]() { return new double[frameSampleCount * inputChannelCount]; }, [](double*& b) { delete[] b; }),
    m_inputEqOutputSpectrumAnalyser(spectrumAnalysisFftLength, sampleFrequency, inputChannelCount,
        spectrumAnalysisPointCountPerDecade)
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
    startInputEqOutputFrameThread();
    startInputEqOutputFrameSpectrumAnalysisThread();
}

void MixerAnalysisDispatcher::stop()
{
    bool wasStopped = m_stopped.load();
    m_stopped.store(true);
    if (!wasStopped)
    {
        stopSoundLevelThread();
        stopInputEqOutputFrameThread();
        stopInputEqOutputFrameSpectrumAnalysisThread();
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

    m_floatInputEqOutputFrameBoundedBuffer.write([&](float*& buffer)
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

    m_doubleInputEqOutputFrameBoundedBuffer.write([&](double*& buffer)
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

void MixerAnalysisDispatcher::floatInputEqOutputFrameRun()
{
    while (!m_stopped.load())
    {
        try
        {
            bool isFinished;
            m_floatInputEqOutputFrameBoundedBuffer.read([&](float* const& inputBuffer)
            {
                m_inputEqOutputSpectrumAnalyser.writePartialData([&](size_t writeIndex, float* analysisBuffer)
                {
                    for (size_t inputIndex = 0; inputIndex < m_inputChannelCount; inputIndex++)
                    {
                        float* destination = analysisBuffer + inputIndex * m_spectrumAnalysisFftLength +
                            writeIndex * m_frameSampleCount;
                        float* source = inputBuffer + inputIndex * m_frameSampleCount;
                        memcpy(destination, source, m_frameSampleCount * sizeof(float));
                    }
                    isFinished = (writeIndex + 1) * m_frameSampleCount >= m_spectrumAnalysisFftLength;
                });
            });

            if (isFinished)
            {
                m_inputEqOutputSpectrumAnalyser.finishWriting();
            }
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void MixerAnalysisDispatcher::doubleInputEqOutputFrameRun()
{
    while (!m_stopped.load())
    {
        try
        {
            bool isFinished;
            m_doubleInputEqOutputFrameBoundedBuffer.read([&](double* const& inputBuffer)
            {
                m_inputEqOutputSpectrumAnalyser.writePartialData([&](size_t writeIndex, float* analysisBuffer)
                {
                    for (size_t inputIndex = 0; inputIndex < m_inputChannelCount; inputIndex++)
                    {
                        float* destination = analysisBuffer + inputIndex * m_spectrumAnalysisFftLength +
                            writeIndex * m_frameSampleCount;
                        double* source = inputBuffer + inputIndex * m_frameSampleCount;
                        for (size_t i = 0; i < m_frameSampleCount; i++)
                        {
                            destination[i] = static_cast<float>(source[i]);
                        }
                    }
                    isFinished = (writeIndex + 1) * m_frameSampleCount >= m_spectrumAnalysisFftLength;
                });

                if (isFinished)
                {
                    m_inputEqOutputSpectrumAnalyser.finishWriting();
                }
            });
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void MixerAnalysisDispatcher::inputEqOutputFrameSpectrumAnalysisRun()
{
    while (!m_stopped.load())
    {
        try
        {
            auto spectrums = m_inputEqOutputSpectrumAnalyser.calculateDecimatedSpectrumAnalysis();
            sendInputSpectrumMessage(spectrums);
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

void MixerAnalysisDispatcher::startInputEqOutputFrameThread()
{
    switch (m_processingDataType)
    {
        case ProcessingDataType::Float:
            m_inputEqOutputFrameThread = make_unique<thread>(&MixerAnalysisDispatcher::floatInputEqOutputFrameRun, this);
            break;

        case ProcessingDataType::Double:
            m_inputEqOutputFrameThread = make_unique<thread>(&MixerAnalysisDispatcher::doubleInputEqOutputFrameRun, this);
            break;

        default:
            THROW_NOT_SUPPORTED_EXCEPTION("Invalid processing data type");
    }
}

void MixerAnalysisDispatcher::startInputEqOutputFrameSpectrumAnalysisThread()
{
    m_inputEqOutputFrameSpectrumAnalysisThread =
        make_unique<thread>(&MixerAnalysisDispatcher::inputEqOutputFrameSpectrumAnalysisRun, this);
}

void MixerAnalysisDispatcher::stopSoundLevelThread()
{
    // Write an empty message to unlock the sound level thread.
    m_soundLevelBoundedBuffer.write([](SoundLevelMessage& m) { });
    m_soundLevelThread->join();
    m_soundLevelThread.release();
}

void MixerAnalysisDispatcher::stopInputEqOutputFrameThread()
{
    // Write an empty message to unlock the input sample thread.
    switch (m_processingDataType)
    {
        case ProcessingDataType::Float:
            m_floatInputEqOutputFrameBoundedBuffer.write([](float*& m) { });
            break;

        case ProcessingDataType::Double:
            m_doubleInputEqOutputFrameBoundedBuffer.write([](double*& m) { });
            break;

        default:
            THROW_NOT_SUPPORTED_EXCEPTION("Invalid processing data type");
    }

    m_inputEqOutputFrameThread->join();
    m_inputEqOutputFrameThread.release();
}

void MixerAnalysisDispatcher::stopInputEqOutputFrameSpectrumAnalysisThread()
{
    // Write an empty message to unlock the sound level thread.
    m_inputEqOutputSpectrumAnalyser.finishWriting();
    m_inputEqOutputFrameSpectrumAnalysisThread->join();
    m_inputEqOutputFrameSpectrumAnalysisThread.release();
}

void MixerAnalysisDispatcher::sendInputSpectrumMessage(const vector<vector<SpectrumPoint>>& spectrums)
{
    vector<ChannelSpectrum> channelSpectrums;
    channelSpectrums.reserve(spectrums.size());

    for (size_t i = 0; i < spectrums.size(); i++)
    {
        channelSpectrums.emplace_back(i, spectrums[i]);
    }

    m_send(InputSpectrumMessage(channelSpectrums));
}
