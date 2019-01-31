#ifndef UTILS_DATA_RAW_AUDIO_FRAME_H
#define UTILS_DATA_RAW_AUDIO_FRAME_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace adaptone
{
    /*
     * A Raw PCM audio frame (Little endian)
     */
    class RawAudioFrame
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
        static RawAudioFrame::Format parseFormat(const std::string& format);

    private:
        Format m_format;
        std::size_t m_channelCount;
        std::size_t m_sampleCount;
        uint8_t* m_data;

    public:
        RawAudioFrame(Format format, std::size_t channelCount, std::size_t sampleCount);
        RawAudioFrame(const RawAudioFrame& other);
        RawAudioFrame(RawAudioFrame&& other);
        ~RawAudioFrame();

        Format format() const;
        std::size_t channelCount() const;
        std::size_t sampleCount() const;

        uint8_t* data();
        std::size_t size() const;

        RawAudioFrame& operator=(const RawAudioFrame& other);
        RawAudioFrame& operator=(RawAudioFrame&& other);
    };

    inline std::size_t RawAudioFrame::formatSize(Format format)
    {
        return static_cast<std::size_t>(format) & 0b0111;
    }

    inline RawAudioFrame::Format RawAudioFrame::format() const
    {
        return m_format;
    }

    inline std::size_t RawAudioFrame::channelCount() const
    {
        return m_channelCount;
    }

    inline std::size_t RawAudioFrame::sampleCount() const
    {
        return m_sampleCount;
    }

    inline uint8_t* RawAudioFrame::data()
    {
        return m_data;
    }

    inline std::size_t RawAudioFrame::size() const
    {
        return m_channelCount * m_sampleCount * formatSize(m_format);
    }
}

#endif