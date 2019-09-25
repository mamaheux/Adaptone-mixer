#include <Mixer/Configuration/AudioInputConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

AudioInputConfiguration::AudioInputConfiguration(const Properties& properties)
{
    constexpr const char* TypePropertyKey = "audio.input.type";
    constexpr const char* FormatPropertyKey = "audio.input.format";

    constexpr const char* InputFilenamePropertyKey = "audio.input.filename";
    constexpr const char* LoopingPropertyKey = "audio.input.looping";

    constexpr const char* DevicePropertyKey = "audio.input.device";

    string type = properties.get<string>(TypePropertyKey);

    m_format = parseFormat(properties.get<string>(FormatPropertyKey));

    if (type == "raw_file")
    {
        m_type = AudioInputConfiguration::Type::RawFile;
        m_filename = properties.get<string>(InputFilenamePropertyKey);
        m_looping = properties.get<bool>(LoopingPropertyKey);
    }
#if defined(__unix__) || defined(__linux__)
    else if (type == "alsa")
    {
        m_type = AudioInputConfiguration::Type::Alsa;
        m_device = properties.get<string>(DevicePropertyKey);
    }
#endif
    else
    {
        THROW_INVALID_VALUE_EXCEPTION(TypePropertyKey, type);
    }
}

AudioInputConfiguration::~AudioInputConfiguration()
{
}
