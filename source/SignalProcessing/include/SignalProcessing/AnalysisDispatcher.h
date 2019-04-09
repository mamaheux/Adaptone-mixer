#ifndef SIGNAL_PROCESSING_ANALYSIS_DISPATCHER_H
#define SIGNAL_PROCESSING_ANALYSIS_DISPATCHER_H

#include <Utils/ClassMacro.h>

#include <vector>
#include <map>

namespace adaptone
{
    class AnalysisDispatcher
    {
    public:
        enum class SoundLevelType
        {
            InputGain,
            InputEq,
            OutputGain
        };

        DECLARE_NOT_COPYABLE(AnalysisDispatcher);
        DECLARE_NOT_MOVABLE(AnalysisDispatcher);

        AnalysisDispatcher();
        virtual ~AnalysisDispatcher();

        virtual void notifySoundLevel(const std::map<SoundLevelType, std::vector<float>>& soundLevels) = 0;
        virtual void notifySoundLevel(const std::map<SoundLevelType, std::vector<double>>& soundLevels) = 0;
    };
}

#endif
