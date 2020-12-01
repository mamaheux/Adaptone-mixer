#include <Uniformization/SignalOverride/HeadphoneProbeSignalOverride.h>

#include <algorithm>

using namespace adaptone;
using namespace std;

constexpr size_t HeadphoneProbeSignalOverride::DataFrameCount;

HeadphoneProbeSignalOverride::HeadphoneProbeSignalOverride(PcmAudioFrameFormat format,
    size_t channelCount,
    size_t frameSampleCount,
    vector<size_t> headphoneChannelIndexes) :
    m_frameSampleCount(frameSampleCount),
    m_headphoneChannelIndexes(move(headphoneChannelIndexes)),
    m_frame(format, channelCount, frameSampleCount),
    m_data(formatSize(format) * frameSampleCount * DataFrameCount),
    m_currentOverrideDataIndex(0),
    m_currentWriteDataIndex(0),
    m_currentProbeId(-1)
{
}

HeadphoneProbeSignalOverride::~HeadphoneProbeSignalOverride()
{
}

const PcmAudioFrame& HeadphoneProbeSignalOverride::override(const PcmAudioFrame& frame)
{
    m_frame = frame;
    uint8_t* probeData;

    {
        lock_guard lock(m_currentOverrideDataIndexMutex);
        probeData = m_data.data() + m_currentOverrideDataIndex;
        m_currentOverrideDataIndex += formatSize(m_frame.format()) * m_frameSampleCount;
        m_currentOverrideDataIndex %= m_data.size();
    }

    PcmAudioFrame probeFrame(m_frame.format(), 1, m_frameSampleCount, probeData);
    for (size_t headphoneChannelIndex : m_headphoneChannelIndexes)
    {
        m_frame.writeChannel(headphoneChannelIndex, probeFrame, 0);
    }

    return m_frame;
}

void HeadphoneProbeSignalOverride::writeData(const ProbeSoundDataMessage& message, uint32_t probeId)
{
    lock_guard lock(m_writeDataMutex);
    if (probeId != m_currentProbeId)
    {
        return;
    }

    const uint8_t* data = message.data();
    size_t dataSize = message.dataSize();

    while (dataSize > 0)
    {
        size_t dataSizeToWrite = min(m_data.size() - m_currentWriteDataIndex, dataSize);
        memcpy(m_data.data() + m_currentWriteDataIndex, data, dataSizeToWrite);

        data += dataSizeToWrite;
        dataSize -= dataSizeToWrite;

        m_currentWriteDataIndex += dataSizeToWrite;
        m_currentWriteDataIndex %= m_data.size();
    }
}
