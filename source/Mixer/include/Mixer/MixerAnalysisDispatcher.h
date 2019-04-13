#ifndef MIXER_MIXER_ANALYSIS_DISPATCHER_H
#define MIXER_MIXER_ANALYSIS_DISPATCHER_H

#include <SignalProcessing/AnalysisDispatcher.h>

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/ClassMacro.h>

#include <vector>
#include <map>
#include <functional>

namespace adaptone
{
    class MixerAnalysisDispatcher : public AnalysisDispatcher
    {
        std::function<void(const ApplicationMessage&)> m_send;

    public:
        MixerAnalysisDispatcher(std::function<void(const ApplicationMessage&)> send);
        ~MixerAnalysisDispatcher() override;

        DECLARE_NOT_COPYABLE(MixerAnalysisDispatcher);
        DECLARE_NOT_MOVABLE(MixerAnalysisDispatcher);

        void notifySoundLevel(const std::map<SoundLevelType, std::vector<float>>& soundLevels) override;
        void notifySoundLevel(const std::map<SoundLevelType, std::vector<double>>& soundLevels) override;
    };
}

#endif
