#ifndef UNIFORMIZATION_MODEL_SIGNAL_OVERRIDE_SWEEP_SIGNAL_OVERRIDE_H
#define UNIFORMIZATION_MODEL_SIGNAL_OVERRIDE_SWEEP_SIGNAL_OVERRIDE_H

#include <Uniformization/SignalOverride/SpecificSignalOverride.h>

#include <armadillo>

#include <atomic>

namespace adaptone
{
    class SweepSignalOverride : public SpecificSignalOverride
    {
        std::size_t m_outputChannelIndex;
        std::size_t m_currentSweepFrame;
        std::atomic<bool> m_sweepActive;

        arma::vec m_sweepVec;
        PcmAudioFrame m_frame;
        PcmAudioFrame m_sweepPcmAudioFrame;

    public:
        SweepSignalOverride(PcmAudioFrameFormat format, size_t sampleFrequency, size_t outputChannelCount,
            size_t frameSampleCount, double f1, double f2, double period);
        ~SweepSignalOverride() override;

        DECLARE_NOT_COPYABLE(SweepSignalOverride);
        DECLARE_NOT_MOVABLE(SweepSignalOverride);

        const PcmAudioFrame& override(const PcmAudioFrame& frame) override;

        void startSweep(std::size_t outputChannelIndex);
        bool isSweepActive();
        const arma::vec& sweepVec() const;
    };

    inline void SweepSignalOverride::startSweep(std::size_t outputChannelIndex)
    {
        m_outputChannelIndex = outputChannelIndex;
        m_currentSweepFrame = 0;
        m_sweepActive.store(true);
    }

    inline bool SweepSignalOverride::isSweepActive()
    {
        return m_sweepActive.load();
    }

    inline const arma::vec& SweepSignalOverride::sweepVec() const
    {
        return m_sweepVec;
    }
}

#endif
