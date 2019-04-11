#include <Mixer/MixerAnalysisDispatcher.h>

using namespace adaptone;
using namespace std;

MixerAnalysisDispatcher::MixerAnalysisDispatcher()
{
}

MixerAnalysisDispatcher::~MixerAnalysisDispatcher()
{
}

void MixerAnalysisDispatcher::notifySoundLevel(const map<SoundLevelType, vector<float>>& soundLevels)
{
}

void MixerAnalysisDispatcher::notifySoundLevel(const map<SoundLevelType, vector<double>>& soundLevels)
{
}
