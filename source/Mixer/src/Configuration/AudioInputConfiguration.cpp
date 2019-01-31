#include <Mixer/Configuration/AudioInputConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

AudioInputConfiguration::AudioInputConfiguration(const Properties& properties)
{
    constexpr const char* TypePropertyKey = "input.type";
    constexpr const char* FormatPropertyKey = "input.format";
    constexpr const char* InputFilenamePropertyKey = "input.filename";

    string type = properties.get<string>(TypePropertyKey);

    m_format = RawAudioFrame::parseFormat(properties.get<string>(FormatPropertyKey));

    if (type == "raw_file")
    {
        m_type = AudioInputConfiguration::Type::RawFile;
        m_filename = properties.get<string>(InputFilenamePropertyKey);
    }
    else
    {
        THROW_INVALID_VALUE_EXCEPTION(TypePropertyKey, type);
    }
}

AudioInputConfiguration::~AudioInputConfiguration()
{
}