#include <Mixer/ChannelIdMapping.h>

#include <Utils/Exception/InvalidValueException.h>

#include <algorithm>
#include <mutex>

using namespace adaptone;
using namespace std;

ChannelIdMapping::ChannelIdMapping(size_t inputChannelCount,
    size_t outputChannelCount,
    const vector<size_t>& headphoneChannelIndexes) :
    m_inputChannelCount(inputChannelCount),
    m_outputChannelCount(outputChannelCount),
    m_headphoneChannelIndexes(headphoneChannelIndexes)
{
}

ChannelIdMapping::~ChannelIdMapping()
{
}

void ChannelIdMapping::update(vector<size_t> inputChannelId, vector<size_t> auxiliaryChannelId, size_t speakerCount)
{
    if (inputChannelId.size() > m_inputChannelCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("inputChannelId", "");
    }
    if (auxiliaryChannelId.size() + speakerCount + m_headphoneChannelIndexes.size() > m_outputChannelCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("auxiliaryChannelId, speakerCount", "");
    }

    unique_lock lock(m_mutex);

    m_inputIndexByChannelId.clear();
    m_auxiliaryOutputIndexByChannelId.clear();
    m_channelIdByInputIndex.clear();
    m_channelIdByAuxiliaryOutputIndex.clear();

    for (size_t i = 0; i < inputChannelId.size(); i++)
    {
        m_inputIndexByChannelId[inputChannelId[i]] = i;
        m_channelIdByInputIndex[i] = inputChannelId[i];
    }

    size_t j = 0;
    for (size_t i = 0; m_auxiliaryOutputIndexByChannelId.size() < auxiliaryChannelId.size(); i++)
    {
        size_t index = m_outputChannelCount - i - 1;
        if (!isHeadphoneChannelIndex(index))
        {
            m_auxiliaryOutputIndexByChannelId[auxiliaryChannelId[j]] = index;
            m_channelIdByAuxiliaryOutputIndex[index] = auxiliaryChannelId[j];
            j++;
        }
    }

    m_masterOutputIndexes.clear();
    for (size_t i = 0; m_masterOutputIndexes.size() < speakerCount; i++)
    {
        if (!isHeadphoneChannelIndex(i))
        {
            m_masterOutputIndexes.push_back(i);
        }
    }
}

bool ChannelIdMapping::isHeadphoneChannelIndex(size_t channelIndex)
{
    auto it = find(m_headphoneChannelIndexes.begin(), m_headphoneChannelIndexes.end(), channelIndex);
    return it != m_headphoneChannelIndexes.end();
}
