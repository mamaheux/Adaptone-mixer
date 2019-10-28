#include <Uniformization/SignalOverride/SweepSignalOverride.h>
#include <Uniformization/Math.h>

#include <armadillo>

using namespace adaptone;

SweepSignalOverride::SweepSignalOverride(PcmAudioFrameFormat format,
    size_t sampleFrequency,
    size_t outputChannelCount,
    size_t frameSampleCount,
    double f1,
    double f2,
    double period) :
    m_sweepPcmAudioFrame(format, 1, frameSampleCount),
    m_frame(format, outputChannelCount, frameSampleCount)
{
    m_sweepVec = logSinChirp<arma::vec>(f1, f2, period, sampleFrequency);

    constexpr size_t ChannelCount = 1;
    AudioFrame<double> sweepAudioFrame(ChannelCount, m_sweepVec.n_elem, m_sweepVec.memptr());

    PcmAudioFrame sweepPcmAudioFrame(sweepAudioFrame, format);
    m_sweepPcmAudioFrame = sweepPcmAudioFrame;

    m_outputChannelIndex = 0;
    m_currentSweepFrame = 0;
    m_sweepActive = false;
}

SweepSignalOverride::~SweepSignalOverride()
{
}

const PcmAudioFrame& SweepSignalOverride::override(const PcmAudioFrame& frame)
{
    m_frame.clear();

    if (m_sweepActive.load())
    {
        size_t offset = m_currentSweepFrame * frame.sampleCount() * formatSize(frame.format());
        m_currentSweepFrame++;

        if (m_currentSweepFrame * frame.sampleCount() > m_sweepVec.size())
        {
            m_sweepActive.store(false);
        }
        else
        {
            constexpr size_t ChannelCount = 1;
            PcmAudioFrame tmp(m_sweepPcmAudioFrame.format(), ChannelCount, frame.sampleCount(),
                m_sweepPcmAudioFrame.data() + offset);
            m_frame.writeChannel(m_outputChannelIndex, tmp, 0);
        }
    }

    return m_frame;
}


