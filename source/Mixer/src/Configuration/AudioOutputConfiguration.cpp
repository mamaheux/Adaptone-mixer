#include <Mixer/Configuration/AudioOutputConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

AudioOutputConfiguration::AudioOutputConfiguration(const Properties& properties)
{
    constexpr const char* TypePropertyKey = "audio.output.type";
    constexpr const char* FormatPropertyKey = "audio.output.format";

    constexpr const char* FilenamePropertyKey = "audio.output.filename";

    constexpr const char* DevicePropertyKey = "audio.output.device";

    constexpr const char* HardwareDelayPropertyKey = "audio.output.hardware_delay";

    string type = properties.get<string>(TypePropertyKey);

    m_format = parseFormat(properties.get<string>(FormatPropertyKey));
    m_hardwareDelay = properties.get<double>(HardwareDelayPropertyKey);

    if (type == "raw_file")
    {
        m_type = AudioOutputConfiguration::Type::RawFile;
        m_filename = properties.get<string>(FilenamePropertyKey);
    }
#if defined(__unix__) || defined(__linux__)
    else if (type == "alsa")
    {
        m_type = AudioOutputConfiguration::Type::Alsa;
        m_device = properties.get<string>(DevicePropertyKey);
    }
#endif
    else
    {
        THROW_INVALID_VALUE_EXCEPTION(TypePropertyKey, type);
    }
}

AudioOutputConfiguration::~AudioOutputConfiguration()
{
}
