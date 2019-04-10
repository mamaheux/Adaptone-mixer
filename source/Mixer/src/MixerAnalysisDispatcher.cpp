#include <Mixer/MixerAnalysisDispatcher.h>

using namespace adaptone;

MixerAnalysisDispatcher::MixerAnalysisDispatcher()
{
}

MixerAnalysisDispatcher::~MixerAnalysisDispatcher()
{
}

void MixerAnalysisDispatcher::notifySoundLevel(const std::map<SoundLevelType, std::vector<float>>& soundLevels)
{
}

void MixerAnalysisDispatcher::notifySoundLevel(const std::map<SoundLevelType, std::vector<double>>& soundLevels)
{
}
