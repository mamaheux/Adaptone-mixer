#if defined(__unix__) || defined(__linux__)

#include <MixerTests/JetsonTest.h>

#include <Mixer/Audio/Alsa/AlsaPcmDevice.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

JETSON_TEST(AlsaPcmDeviceTests, playback_shouldPlayASound)
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
        cout << i << endl;
        if (captureDevice.read(frame))
        {
            playbackDevice.write(frame);
        }
    }
}

#endif
