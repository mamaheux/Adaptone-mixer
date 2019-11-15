#ifndef UNIFORMIZATION_SIGNAL_OVERRIDE_HEADPHONE_PROBE_SIGNAL_OVERRIDE_H
#define UNIFORMIZATION_SIGNAL_OVERRIDE_HEADPHONE_PROBE_SIGNAL_OVERRIDE_H

#include <Uniformization/SignalOverride/SpecificSignalOverride.h>
#include <Uniformization/Communication/Messages/Udp/ProbeSoundDataMessage.h>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <iostream>

namespace adaptone
{
    class HeadphoneProbeSignalOverride : public SpecificSignalOverride
    {
        static constexpr std::size_t DataFrameCount = 100;

        std::size_t m_frameSampleCount;
        std::vector<std::size_t> m_headphoneChannelIndexes;

        std::vector<uint8_t> m_data;
        PcmAudioFrame m_frame;
        std::mutex m_currentOverrideDataIndexMutex;
        std::mutex m_writeDataMutex;

        std::size_t m_currentOverrideDataIndex;
        std::size_t m_currentWriteDataIndex;

        uint32_t m_currentProbeId;

        uint16_t m_lastSoundDataId;
        bool m_isFirstMessage;

    public:
        HeadphoneProbeSignalOverride(PcmAudioFrameFormat format,
            std::size_t channelCount,
            std::size_t frameSampleCount,
            const std::vector<std::size_t>& headphoneChannelIndexes);
        ~HeadphoneProbeSignalOverride() override;

        DECLARE_NOT_COPYABLE(HeadphoneProbeSignalOverride);
        DECLARE_NOT_MOVABLE(HeadphoneProbeSignalOverride);

        const PcmAudioFrame& override(const PcmAudioFrame& frame) override;
        void writeData(const ProbeSoundDataMessage& message, uint32_t probeId);

        void setCurrentProbeId(uint32_t currentProbeId);
    };

    inline void HeadphoneProbeSignalOverride::setCurrentProbeId(uint32_t currentProbeId)
    {
        std::cout << "listenToProbeSound : " << std::endl;
        std::lock_guard currentOverrideDataIndexLock(m_currentOverrideDataIndexMutex);
        std::lock_guard currentWriteDataIndexLock(m_writeDataMutex);
        m_currentOverrideDataIndex = 0;
        m_currentWriteDataIndex = 0;

        m_lastSoundDataId = 0;
        m_isFirstMessage = true;

        m_currentProbeId = currentProbeId;
    }
}

#endif
