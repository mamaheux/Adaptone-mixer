#ifndef UNIFORMIZATION_COMMUNICATION_RECORD_RESPONSE_MESSAGE_AGREGATOR_H
#define UNIFORMIZATION_COMMUNICATION_RECORD_RESPONSE_MESSAGE_AGREGATOR_H

#include <Uniformization/Communication/Messages/Tcp/RecordResponseMessage.h>

#include <Utils/Data/AudioFrame.h>
#include <Utils/Data/PcmAudioFrameFormat.h>

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

namespace adaptone
{
    class RecordResponseMessageAgregator
    {
        PcmAudioFrameFormat m_format;

        std::mutex m_dataMutex;
        bool m_stopped;
        uint8_t m_currentRecordId;
        std::size_t m_probeCount;
        std::unordered_map<uint32_t, AudioFrame<double>> m_framesByProbeId;

        std::condition_variable m_conditionVariable;
        std::mutex m_conditionVariableMutex;

    public:
        RecordResponseMessageAgregator(PcmAudioFrameFormat format);
        virtual ~RecordResponseMessageAgregator();

        void reset(uint8_t currentRecordId, std::size_t probeCount);
        void agregate(const RecordResponseMessage& message, uint32_t probeId);
        std::optional<const std::unordered_map<uint32_t, AudioFrame<double>>> read(int timeoutMs);
    };
}

#endif
