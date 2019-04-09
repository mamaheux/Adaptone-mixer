#ifndef MIXER_MIXER_ANALYSIS_DISPATCHER_H
#define MIXER_MIXER_ANALYSIS_DISPATCHER_H

#include <SignalProcessing/AnalysisDispatcher.h>

#include <Utils/ClassMacro.h>

#include <vector>
#include <map>

namespace adaptone
{
    class MixerAnalysisDispatcher : public AnalysisDispatcher
    {
    public:
        MixerAnalysisDispatcher();
        ~MixerAnalysisDispatcher() override;

        DECLARE_NOT_COPYABLE(MixerAnalysisDispatcher);
        DECLARE_NOT_MOVABLE(MixerAnalysisDispatcher);

        void notifySoundLevel(const std::map<SoundLevelType, std::vector<float>>& soundLevels) override;
        void notifySoundLevel(const std::map<SoundLevelType, std::vector<double>>& soundLevels) override;
    };
}

#endif
