#include <Uniformization/Communication/RecordResponseMessageAgregator.h>

#include <Utils/Data/PcmAudioFrame.h>

using namespace adaptone;
using namespace std;

RecordResponseMessageAgregator::RecordResponseMessageAgregator(PcmAudioFrameFormat format) :
    m_format(format)
{
}

RecordResponseMessageAgregator::~RecordResponseMessageAgregator()
{
}

void RecordResponseMessageAgregator::reset(uint8_t currentRecordId, size_t probeCount)
{
    lock_guard lock(m_dataMutex);
    m_stopped = false;
    m_currentRecordId = currentRecordId;
    m_probeCount = probeCount;
    m_framesByProbeId.clear();
}

void RecordResponseMessageAgregator::agregate(const RecordResponseMessage& message, uint32_t probeId)
{
    lock_guard lock(m_dataMutex);
    if (m_stopped || m_currentRecordId != message.recordId())
    {
        return;
    }

    constexpr size_t ChannelCount = 1;
    size_t sampleCount = message.dataSize() / formatSize(m_format);
    m_framesByProbeId.emplace(probeId,
        PcmAudioFrame(m_format, ChannelCount, sampleCount, const_cast<uint8_t*>(message.data())));

    if (m_framesByProbeId.size() == m_probeCount)
    {
        m_stopped = true;
        m_conditionVariable.notify_all();
    }
}

optional<unordered_map<uint32_t, AudioFrame<double>>> RecordResponseMessageAgregator::read(int timeoutMs)
{
    {
        lock_guard lock(m_dataMutex);
        if (m_framesByProbeId.size() == m_probeCount)
        {
            return m_framesByProbeId;
        }
    }

    unique_lock conditionVariableLock(m_conditionVariableMutex);
    bool timeout = m_conditionVariable.wait_for(conditionVariableLock, chrono::milliseconds(timeoutMs),
        [&]()
        {
            lock_guard lock(m_dataMutex);
            return m_framesByProbeId.size() == m_probeCount;
        });

    if (timeout)
    {
        return m_framesByProbeId;
    }
    else
    {
        return nullopt;
    }
}
