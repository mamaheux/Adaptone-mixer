#include <Mixer/Configuration/AudioConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

AudioConfiguration::AudioConfiguration(const Properties& properties)
{
    constexpr const char* FrameSampleCountPropertyKey = "audio.frame_sample_count";
    constexpr const char* SampleFrequencyPropertyKey = "audio.sample_frequency";
    constexpr const char* InputChannelCountPropertyKey = "audio.input_channel_count";
    constexpr const char* OutputChannelCountPropertyKey = "audio.output_channel_count";

    constexpr const char* EqCenterFrequenciesPropertyKey = "audio.eq.center_frequencies";

    constexpr const char* SoundLevelLengthPropertyKey = "audio.analysis.sound_level_length";
    constexpr const char* SpectrumAnalysisPointCountPerDecadePropertyKey =
        "audio.analysis.spectrum.point_count_per_decade";

    m_frameSampleCount = properties.get<size_t>(FrameSampleCountPropertyKey);
    m_sampleFrequency = properties.get<size_t>(SampleFrequencyPropertyKey);
    m_inputChannelCount = properties.get<size_t>(InputChannelCountPropertyKey);
    m_outputChannelCount = properties.get<size_t>(OutputChannelCountPropertyKey);

    setProcessingDataType(properties);

    m_eqCenterFrequencies = properties.get<vector<double>>(EqCenterFrequenciesPropertyKey);

    setMaxOutputDelay(properties);

    m_soundLevelLength = properties.get<size_t>(SoundLevelLengthPropertyKey);
    m_spectrumAnalysisPointCountPerDecade = properties.get<size_t>(SpectrumAnalysisPointCountPerDecadePropertyKey);
    setSpectrumAnalysisFftLength(properties);

    setHeadphoneChannelIndexes(properties);
}

AudioConfiguration::~AudioConfiguration()
{
}

void AudioConfiguration::setProcessingDataType(const Properties& properties)
{
    constexpr const char* ProcessingDataTypePropertyKey = "audio.processing_data_type";

    string processingDataType = properties.get<string>(ProcessingDataTypePropertyKey);
    if (processingDataType == "float")
    {
        m_processingDataType = ProcessingDataType::Float;
    }
    else if (processingDataType == "double")
    {
        m_processingDataType = ProcessingDataType::Double;
    }
    else
    {
        THROW_INVALID_VALUE_EXCEPTION(ProcessingDataTypePropertyKey, processingDataType);
    }
}

void AudioConfiguration::setMaxOutputDelay(const Properties& properties)
{
    constexpr const char* MaxOutputDelayPropertyKey = "audio.max_output_delay";

    m_maxOutputDelay = properties.get<size_t>(MaxOutputDelayPropertyKey);
    if (m_maxOutputDelay % m_frameSampleCount != 0)
    {
        THROW_INVALID_VALUE_EXCEPTION(MaxOutputDelayPropertyKey,
            "Must be a multiple of audio.frame_sample_count");
    }
}

void AudioConfiguration::setSpectrumAnalysisFftLength(const Properties& properties)
{
    constexpr const char* SpectrumAnalysisFftLengthPropertyKey = "audio.analysis.spectrum.fft_length";

    m_spectrumAnalysisFftLength = properties.get<size_t>(SpectrumAnalysisFftLengthPropertyKey);
    if (m_spectrumAnalysisFftLength % m_frameSampleCount != 0)
    {
        THROW_INVALID_VALUE_EXCEPTION(SpectrumAnalysisFftLengthPropertyKey,
            "Must be a multiple of audio.frame_sample_count");
    }
}

void AudioConfiguration::setHeadphoneChannelIndexes(const Properties& properties)
{
    constexpr const char* HeadphoneChannelIndexPropertyKey = "audio.headphone_channel_indexes";

    m_headphoneChannelIndexes = properties.get<vector<size_t>>(HeadphoneChannelIndexPropertyKey);
    if (m_headphoneChannelIndexes.size() != 1 && m_headphoneChannelIndexes.size() != 2)
    {
        THROW_INVALID_VALUE_EXCEPTION(HeadphoneChannelIndexPropertyKey, "The size must be 1 or 2");
    }
    for (size_t index : m_headphoneChannelIndexes)
    {
        if (index >= m_outputChannelCount)
        {
            THROW_INVALID_VALUE_EXCEPTION(HeadphoneChannelIndexPropertyKey, "The indexes must be valid");
        }
    }
}
