#ifndef SIGNAL_PROCESSING_ANALYSIS_DISPATCHER_H
#define SIGNAL_PROCESSING_ANALYSIS_DISPATCHER_H

#include <Utils/ClassMacro.h>

#include <vector>
#include <map>
#include <functional>

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

        AnalysisDispatcher();
        virtual ~AnalysisDispatcher();

        DECLARE_NOT_COPYABLE(AnalysisDispatcher);
        DECLARE_NOT_MOVABLE(AnalysisDispatcher);

        virtual void start() = 0;
        virtual void stop() = 0;

        virtual void notifySoundLevel(const std::map<SoundLevelType, std::vector<float>>& soundLevels) = 0;
        virtual void notifySoundLevel(const std::map<SoundLevelType, std::vector<double>>& soundLevels) = 0;

        virtual void notifyInputEqOutputFrame(const std::function<void(float*)> notifyFunction) = 0;
        virtual void notifyInputEqOutputFrame(const std::function<void(double*)> notifyFunction) = 0;
    };
}

#endif
