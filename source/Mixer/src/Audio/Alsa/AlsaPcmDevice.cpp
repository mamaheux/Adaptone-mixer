#if defined(__unix__) || defined(__linux__)

#include <Mixer/Audio/Alsa/AlsaPcmDevice.h>

#include <Mixer/Audio/Alsa/AlsaException.h>

#include <Utils/Exception/NotSupportedException.h>
#include <Utils/Exception/InvalidValueException.h>

#include <map>

using namespace adaptone;
using namespace std;

class PcmParamsDeleter
{
public:
    void operator()(snd_pcm_hw_params_t* handle)
    {
        snd_pcm_hw_params_free(handle);
    }
};

AlsaPcmDevice::AlsaPcmDevice(const std::string& device,
    AlsaPcmDevice::Stream stream,
    PcmAudioFrame::Format format,
    std::size_t channelCount,
    std::size_t frameSampleCount,
    std::size_t sampleFrequency) : m_format(format), m_channelCount(channelCount), m_frameSampleCount(frameSampleCount)
{
    int err;
    snd_pcm_t* pcmHandlePointer;
    snd_pcm_hw_params_t* paramsPointer;

    if ((err = snd_pcm_open(&pcmHandlePointer, device.c_str(), static_cast<snd_pcm_stream_t>(stream), 0)) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot open audio device: " + device, err, snd_strerror(err));
    }

    std::unique_ptr<snd_pcm_t, PcmDeleter> pcmHandle(pcmHandlePointer);

    if ((err = snd_pcm_hw_params_malloc(&paramsPointer)) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot allocate hardware parameter structure", err, snd_strerror(err));
    }

    std::unique_ptr<snd_pcm_hw_params_t, PcmParamsDeleter> params(paramsPointer);

    if ((err = snd_pcm_hw_params_any(pcmHandle.get(), params.get())) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot initialize hardware parameter structure", err, snd_strerror(err));
    }

    if ((err = snd_pcm_hw_params_set_access(pcmHandle.get(), params.get(), SND_PCM_ACCESS_RW_INTERLEAVED)) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot set access type", err, snd_strerror(err));
    }

    if ((err = snd_pcm_hw_params_set_format(pcmHandle.get(), params.get(), convert(format))) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot set sample format", err, snd_strerror(err));
    }

    if ((err = snd_pcm_hw_params_set_rate(pcmHandle.get(), params.get(), static_cast<int>(sampleFrequency), 0)) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot set sample rate", err, snd_strerror(err));
    }

    if ((err = snd_pcm_hw_params_set_channels(pcmHandle.get(), params.get(), static_cast<int>(channelCount))) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot set channel count", err, snd_strerror(err));
    }

    unsigned int periodSize = static_cast<int>(frameSampleCount);
    if ((err = snd_pcm_hw_params_set_period_size(pcmHandle.get(), params.get(), periodSize, 0)) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot set period size", err, snd_strerror(err));
    }

    if ((err = snd_pcm_hw_params(pcmHandle.get(), params.get())) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot set parameters", err, snd_strerror(err));
    }

    if ((err = snd_pcm_prepare(pcmHandle.get())) < 0)
    {
        THROW_ALSA_EXCEPTION("Cannot prepare audio interface for use", err, snd_strerror(err));
    }

    m_pcmHandle = move(pcmHandle);
}

AlsaPcmDevice::~AlsaPcmDevice()
{

}

bool AlsaPcmDevice::read(PcmAudioFrame& frame)
{
    if (frame.format() != m_format ||
        frame.channelCount() != m_channelCount ||
        frame.sampleCount() != m_frameSampleCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("format, channelCount, sampleCount", "");
    }

    snd_pcm_uframes_t periodSize = static_cast<snd_pcm_uframes_t>(m_frameSampleCount);
    int err = snd_pcm_readi(m_pcmHandle.get(), &frame[0], periodSize);
    if (err == -EPIPE)
    {
        // EPIPE means overrun
        snd_pcm_prepare(m_pcmHandle.get());
        return false;
    }
    else if (err != periodSize)
    {
        THROW_ALSA_EXCEPTION("Read from audio interface failed", err, snd_strerror(err));
    }
    return true;
}

void AlsaPcmDevice::write(const PcmAudioFrame& frame)
{
    if (frame.format() != m_format ||
        frame.channelCount() != m_channelCount ||
        frame.sampleCount() != m_frameSampleCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("format, channelCount, sampleCount", "");
    }

    snd_pcm_uframes_t periodSize = static_cast<snd_pcm_uframes_t>(m_frameSampleCount);
    int err = snd_pcm_writei(m_pcmHandle.get(), frame.data(), periodSize);

    if (err == -EPIPE)
    {
        // EPIPE means underrun
        err = snd_pcm_recover(m_pcmHandle.get(), err, 1);
        if (err >= 0)
        {
            err = snd_pcm_writei(m_pcmHandle.get(), frame.data(), periodSize);
        }
    }

    if (err != periodSize)
    {
        THROW_ALSA_EXCEPTION("Write to audio interface failed", err, snd_strerror(err));
    }
}

snd_pcm_format_t AlsaPcmDevice::convert(PcmAudioFrame::Format format)
{
    map<PcmAudioFrame::Format, snd_pcm_format_t> Mapping(
        {
            { PcmAudioFrame::Format::Signed8, SND_PCM_FORMAT_S8 },
            { PcmAudioFrame::Format::Signed16, SND_PCM_FORMAT_S16_LE },
            { PcmAudioFrame::Format::Signed24, SND_PCM_FORMAT_S24_LE },
            { PcmAudioFrame::Format::Signed32, SND_PCM_FORMAT_S32_LE },

            { PcmAudioFrame::Format::Unsigned8, SND_PCM_FORMAT_U8 },
            { PcmAudioFrame::Format::Unsigned16, SND_PCM_FORMAT_U16_LE },
            { PcmAudioFrame::Format::Unsigned24, SND_PCM_FORMAT_U24_LE },
            { PcmAudioFrame::Format::Unsigned32, SND_PCM_FORMAT_U32_LE },

            { PcmAudioFrame::Format::Float, SND_PCM_FORMAT_FLOAT_LE },
            { PcmAudioFrame::Format::Double, SND_PCM_FORMAT_FLOAT64_LE }
        });

    auto it = Mapping.find(format);
    if (it != Mapping.end())
    {
        return it->second;
    }

    THROW_NOT_SUPPORTED_EXCEPTION("Not supported format");
}

#endif
