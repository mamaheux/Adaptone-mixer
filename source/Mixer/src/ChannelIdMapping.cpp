#include <Mixer/ChannelIdMapping.h>

#include <Utils/Exception/InvalidValueException.h>

#include <mutex>

using namespace adaptone;
using namespace std;

ChannelIdMapping::ChannelIdMapping(size_t inputChannelCount, size_t outputChannelCount) :
    m_inputChannelCount(inputChannelCount),
    m_outputChannelCount(outputChannelCount)
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
    if (auxiliaryChannelId.size() + speakerCount > m_outputChannelCount)
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

    for (size_t i = 0; i < auxiliaryChannelId.size(); i++)
    {
        size_t index = m_outputChannelCount - i - 1;
        m_auxiliaryOutputIndexByChannelId[auxiliaryChannelId[i]] = index;
        m_channelIdByAuxiliaryOutputIndex[index] = auxiliaryChannelId[i];
    }

    m_masterOutputIndexes.clear();
    for (size_t i = 0; i < speakerCount; i++)
    {
        m_masterOutputIndexes.push_back(i);
    }
}
