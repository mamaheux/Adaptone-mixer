#ifndef MIXER_CHANNEL_ID_MAPPING_H
#define MIXER_CHANNEL_ID_MAPPING_H

#include <Utils/ClassMacro.h>

#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <optional>

namespace adaptone
{
    class ChannelIdMapping
    {
        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;

        std::shared_mutex m_mutex;

        std::unordered_map<std::size_t, std::size_t> m_inputIndexByChannelId;
        std::unordered_map<std::size_t, std::size_t> m_auxiliaryOutputIndexByChannelId;
        std::unordered_map<std::size_t, std::size_t> m_channelIdByInputIndex;
        std::unordered_map<std::size_t, std::size_t> m_channelIdByAuxiliaryOutputIndex;

        std::vector<std::size_t> m_masterOutputIndexes;

    public:
        ChannelIdMapping(std::size_t inputChannelCount, std::size_t outputChannelCount);
        virtual ~ChannelIdMapping();

        DECLARE_NOT_COPYABLE(ChannelIdMapping);
        DECLARE_NOT_MOVABLE(ChannelIdMapping);

        void update(std::vector<std::size_t> inputChannelId,
            std::vector<std::size_t> auxiliaryChannelId,
            std::size_t speakerCount);

        std::size_t getChannelIdFromInputIndex(std::size_t index);
        std::size_t getChannelIdFromAuxiliaryOutputIndex(std::size_t index);
        std::size_t getInputIndexFromChannelId(std::size_t channelId);
        std::size_t getAuxiliaryOutputIndexFromChannelId(std::size_t channelId);

        std::optional<std::size_t> getChannelIdFromInputIndexOrNull(std::size_t index);
        std::optional<std::size_t> getChannelIdFromAuxiliaryOutputIndexOrNull(std::size_t index);
        std::optional<std::size_t> getInputIndexFromChannelIdOrNull(std::size_t channelId);
        std::optional<std::size_t> getAuxiliaryOutputIndexFromChannelIdOrNull(std::size_t channelId);

        std::vector<std::size_t> getMasterOutputIndexes();
        std::size_t getMasterChannelId();
    };

    inline std::size_t ChannelIdMapping::getChannelIdFromInputIndex(std::size_t index)
    {
        std::shared_lock lock(m_mutex);
        return m_channelIdByInputIndex.at(index);
    }

    inline std::size_t ChannelIdMapping::getChannelIdFromAuxiliaryOutputIndex(std::size_t index)
    {
        std::shared_lock lock(m_mutex);
        return m_channelIdByAuxiliaryOutputIndex.at(index);
    }

    inline std::size_t ChannelIdMapping::getInputIndexFromChannelId(std::size_t channelId)
    {
        std::shared_lock lock(m_mutex);
        return m_inputIndexByChannelId.at(channelId);
    }

    inline std::size_t ChannelIdMapping::getAuxiliaryOutputIndexFromChannelId(std::size_t channelId)
    {
        std::shared_lock lock(m_mutex);
        return m_auxiliaryOutputIndexByChannelId.at(channelId);
    }

    inline std::optional<std::size_t> ChannelIdMapping::getChannelIdFromInputIndexOrNull(std::size_t index)
    {
        std::shared_lock lock(m_mutex);
        auto it = m_channelIdByInputIndex.find(index);
        if (it != m_channelIdByInputIndex.end())
        {
            return std::optional<std::size_t>(it->second);
        }
        return std::nullopt;
    }

    inline std::optional<std::size_t> ChannelIdMapping::getChannelIdFromAuxiliaryOutputIndexOrNull(std::size_t index)
    {
        std::shared_lock lock(m_mutex);
        auto it = m_channelIdByAuxiliaryOutputIndex.find(index);
        if (it != m_channelIdByInputIndex.end())
        {
            return std::optional<std::size_t>(it->second);
        }
        return std::nullopt;
    }

    inline std::optional<std::size_t> ChannelIdMapping::getInputIndexFromChannelIdOrNull(std::size_t channelId)
    {
        std::shared_lock lock(m_mutex);
        auto it = m_inputIndexByChannelId.find(channelId);
        if (it != m_channelIdByInputIndex.end())
        {
            return std::optional<std::size_t>(it->second);
        }
        return std::nullopt;
    }

    inline std::optional<std::size_t> ChannelIdMapping::getAuxiliaryOutputIndexFromChannelIdOrNull(std::size_t channelId)
    {
        std::shared_lock lock(m_mutex);
        auto it = m_auxiliaryOutputIndexByChannelId.find(channelId);
        if (it != m_channelIdByInputIndex.end())
        {
            return std::optional<std::size_t>(it->second);
        }
        return std::nullopt;
    }

    inline std::vector<std::size_t> ChannelIdMapping::getMasterOutputIndexes()
    {
        std::shared_lock lock(m_mutex);
        return m_masterOutputIndexes;
    }

    inline std::size_t ChannelIdMapping::getMasterChannelId()
    {
        return 0;
    }
}

#endif
