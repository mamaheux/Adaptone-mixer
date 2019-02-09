#if defined(__unix__) || defined(__linux__)

#include <MixerTests/JetsonTest.h>

#include <Mixer/Audio/Alsa/AlsaPcmDevice.h>
#include <Mixer/AudioInput/AlsaAudioInput.h>
#include <Mixer/AudioOutput/AlsaAudioOutput.h>

using namespace adaptone;
using namespace std;

JETSON_TEST(AlsaTests, captureThenPlay_device_shouldReplayASound)
{
    PcmAudioFrame::Format format = PcmAudioFrame::Format::Signed32;
    std::size_t channelCount = 10;
    std::size_t frameSampleCount = 32;
    std::size_t sampleFrequency = 44100;

    AlsaPcmDevice playbackDevice("hw:CARD=x20,DEV=0",
        AlsaPcmDevice::Stream::Playback,
        format,
        channelCount,
        frameSampleCount,
        sampleFrequency);

    AlsaPcmDevice captureDevice("hw:CARD=x20,DEV=0",
        AlsaPcmDevice::Stream::Capture,
        format,
        channelCount,
        frameSampleCount,
        sampleFrequency);

    PcmAudioFrame frame(format, channelCount, frameSampleCount);

    int frameCount = 10000;
    for (int i = 0; i < frameCount; i++)
    {
        if (captureDevice.read(frame))
        {
            playbackDevice.write(frame);
        }
    }
}

JETSON_TEST(AlsaTests, captureThenPlay_shouldReplayASound)
{
    PcmAudioFrame::Format format = PcmAudioFrame::Format::Signed32;
    std::size_t channelCount = 10;
    std::size_t frameSampleCount = 32;
    std::size_t sampleFrequency = 44100;

    AlsaAudioInput input(format,
        channelCount,
        frameSampleCount,
        sampleFrequency,
        "hw:CARD=x20,DEV=0");

    AlsaAudioOutput output(format,
        channelCount,
        frameSampleCount,
        sampleFrequency,
        "hw:CARD=x20,DEV=0");

    PcmAudioFrame frame(format, channelCount, frameSampleCount);

    int frameCount = 10000;
    for (int i = 0; i < frameCount; i++)
    {
        output.write(input.read());
    }
}

#endif
