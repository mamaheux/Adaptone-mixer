#include <Mixer/Configuration/AudioOutputConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

AudioOutputConfiguration::AudioOutputConfiguration(const Properties& properties)
{
    constexpr const char* TypePropertyKey = "audio.output.type";
    constexpr const char* FormatPropertyKey = "audio.output.format";
    constexpr const char* FilenamePropertyKey = "audio.output.filename";

    string type = properties.get<string>(TypePropertyKey);

    m_format = PcmAudioFrame::parseFormat(properties.get<string>(FormatPropertyKey));

    if (type == "raw_file")
    {
        m_type = AudioOutputConfiguration::Type::RawFile;
        m_filename = properties.get<string>(FilenamePropertyKey);
    }
    else
    {
        THROW_INVALID_VALUE_EXCEPTION(TypePropertyKey, type);
    }
}

AudioOutputConfiguration::~AudioOutputConfiguration()
{
}
