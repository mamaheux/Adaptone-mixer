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
    constexpr const char* ProcessingDataTypePropertyKey = "audio.processing_data_type";

    constexpr const char* EqCenterFrequenciesPropertyKey = "audio.eq.center_frequencies";

    constexpr const char* SoundLevelLengthPropertyKey = "audio.analysis.sound_level_length";
    constexpr const char* SpectrumAnalysisFftLengthPropertyKey = "audio.analysis.spectrum.fft_length";
    constexpr const char* SpectrumAnalysisPointCountPerDecadePropertyKey =
        "audio.analysis.spectrum.point_count_per_decade";

    m_frameSampleCount = properties.get<size_t>(FrameSampleCountPropertyKey);
    m_sampleFrequency = properties.get<size_t>(SampleFrequencyPropertyKey);
    m_inputChannelCount = properties.get<size_t>(InputChannelCountPropertyKey);
    m_outputChannelCount = properties.get<size_t>(OutputChannelCountPropertyKey);

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

    m_eqCenterFrequencies = properties.get<vector<double>>(EqCenterFrequenciesPropertyKey);

    m_soundLevelLength = properties.get<size_t>(SoundLevelLengthPropertyKey);
    m_spectrumAnalysisFftLength = properties.get<size_t>(SpectrumAnalysisFftLengthPropertyKey);
    m_spectrumAnalysisPointCountPerDecade = properties.get<size_t>(SpectrumAnalysisPointCountPerDecadePropertyKey);

    if (m_spectrumAnalysisFftLength % m_frameSampleCount != 0)
    {
        THROW_INVALID_VALUE_EXCEPTION(SpectrumAnalysisFftLengthPropertyKey,
            "Must be a multiple of audio.frame_sample_count");
    }
}

AudioConfiguration::~AudioConfiguration()
{
}
