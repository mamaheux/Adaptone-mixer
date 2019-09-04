#include <Mixer/MixerAnalysisDispatcher.h>

using namespace adaptone;
using namespace std;

constexpr size_t MixerAnalysisDispatcher::SoundLevelBoundedBufferSize;

MixerAnalysisDispatcher::MixerAnalysisDispatcher(shared_ptr<Logger> logger,
    function<void(const ApplicationMessage&)> send) :
    m_logger(logger), m_send(send), m_soundLevelBoundedBuffer(SoundLevelBoundedBufferSize)
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
    m_soundLevelThread = make_unique<thread>(&MixerAnalysisDispatcher::soundLevelRun, this);
}

void MixerAnalysisDispatcher::stop()
{
    bool wasStopped = m_stopped.load();
    m_stopped.store(true);
    if (!wasStopped)
    {
        // Write an empty message to unlock the sound level thread.
        m_soundLevelBoundedBuffer.write([](SoundLevelMessage& m) { });
        m_soundLevelThread->join();
        m_soundLevelThread.release();
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
