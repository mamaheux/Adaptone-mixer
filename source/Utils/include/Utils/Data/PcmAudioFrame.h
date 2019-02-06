#ifndef UTILS_DATA_PCM_AUDIO_FRAME_H
#define UTILS_DATA_PCM_AUDIO_FRAME_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <istream>

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

            Unsigned8 = 1 + 16,
            Unsigned16 = 2 + 16,
            Unsigned24 = 3 + 16,
            UnsignedPadded24 = 4 + 16,
            Unsigned32 = 4 + 16,

            Float = 4 + 32,
            Double = 8 + 32
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

        const uint8_t* data() const;
        std::size_t size() const;

        PcmAudioFrame& operator=(const PcmAudioFrame& other);
        PcmAudioFrame& operator=(PcmAudioFrame&& other);

        uint8_t& operator[](std::size_t i);

        friend std::istream& operator>>(std::istream& stream, PcmAudioFrame& frame);
        friend std::ostream& operator<<(std::ostream& stream, const PcmAudioFrame& frame);
    };

    inline std::size_t PcmAudioFrame::formatSize(Format format)
    {
        return static_cast<std::size_t>(format) & 0b1111;
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

    inline const uint8_t* PcmAudioFrame::data() const
    {
        return m_data;
    }

    inline std::size_t PcmAudioFrame::size() const
    {
        return m_channelCount * m_sampleCount * formatSize(m_format);
    }

    inline uint8_t& PcmAudioFrame::operator[](std::size_t i)
    {
        return m_data[i];
    }

    inline std::istream& operator>>(std::istream& stream, PcmAudioFrame& frame)
    {
        stream.read(reinterpret_cast<char*>(frame.m_data), frame.size());
        return stream;
    }

    inline std::ostream& operator<<(std::ostream& stream, const PcmAudioFrame& frame)
    {
        stream.write(reinterpret_cast<char*>(frame.m_data), frame.size());
        return stream;
    }
}

#endif
