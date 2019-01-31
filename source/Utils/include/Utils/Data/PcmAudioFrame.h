#ifndef UTILS_DATA_PCM_AUDIO_FRAME_H
#define UTILS_DATA_PCM_AUDIO_FRAME_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace adaptone
{
    /*
     * A PCM audio frame (Little endian)
     */
    class PcmAudioFrame
    {
    public:
        enum class Format : std::size_t
        {
            Signed8 = 1,
            Signed16 = 2,
            Signed24 = 3,
            SignedPadded24 = 4,
            Signed32 = 4,
            Unsigned8 = 1 + 8,
            Unsigned16 = 2 + 8,
            Unsigned24 = 3 + 8,
            UnsignedPadded24 = 4 + 8,
            Unsigned32 = 4 + 8
        };

        static std::size_t formatSize(Format format);
        static PcmAudioFrame::Format parseFormat(const std::string& format);

    private:
        Format m_format;
        std::size_t m_channelCount;
        std::size_t m_sampleCount;
        uint8_t* m_data;

    public:
        PcmAudioFrame(Format format, std::size_t channelCount, std::size_t sampleCount);
        PcmAudioFrame(const PcmAudioFrame& other);
        PcmAudioFrame(PcmAudioFrame&& other);
        ~PcmAudioFrame();

        Format format() const;
        std::size_t channelCount() const;
        std::size_t sampleCount() const;

        uint8_t* data();
        std::size_t size() const;

        PcmAudioFrame& operator=(const PcmAudioFrame& other);
        PcmAudioFrame& operator=(PcmAudioFrame&& other);
    };

    inline std::size_t PcmAudioFrame::formatSize(Format format)
    {
        return static_cast<std::size_t>(format) & 0b0111;
    }

    inline PcmAudioFrame::Format PcmAudioFrame::format() const
    {
        return m_format;
    }

    inline std::size_t PcmAudioFrame::channelCount() const
    {
        return m_channelCount;
    }

    inline std::size_t PcmAudioFrame::sampleCount() const
    {
        return m_sampleCount;
    }

    inline uint8_t* PcmAudioFrame::data()
    {
        return m_data;
    }

    inline std::size_t PcmAudioFrame::size() const
    {
        return m_channelCount * m_sampleCount * formatSize(m_format);
    }
}

#endif