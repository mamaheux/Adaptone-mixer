#include <Mixer/MixerAnalysisDispatcher.h>

#include <Communication/Messages/Output/SoundLevelMessage.h>

using namespace adaptone;
using namespace std;

MixerAnalysisDispatcher::MixerAnalysisDispatcher(function<void(const ApplicationMessage&)> send) :
    m_send(send)
{
}

MixerAnalysisDispatcher::~MixerAnalysisDispatcher()
{
}

void MixerAnalysisDispatcher::notifySoundLevel(const map<SoundLevelType, vector<float>>& soundLevels)
{
    //TODO Ne pas envoyer le message sur le thread principal
    const vector<float>& inputAfterGain = soundLevels.at(SoundLevelType::InputGain);
    const vector<float>& inputAfterEq = soundLevels.at(SoundLevelType::InputEq);
    const vector<float>& outputAfterGain = soundLevels.at(SoundLevelType::OutputGain);

    m_send(SoundLevelMessage(vector<double>(inputAfterGain.cbegin(), inputAfterGain.cend()),
        vector<double>(inputAfterEq.cbegin(), inputAfterEq.cend()),
        vector<double>(outputAfterGain.cbegin(), outputAfterGain.cend())));
}

void MixerAnalysisDispatcher::notifySoundLevel(const map<SoundLevelType, vector<double>>& soundLevels)
{
    //TODO Ne pas envoyer le message sur le thread principal
    m_send(SoundLevelMessage(soundLevels.at(SoundLevelType::InputGain),
        soundLevels.at(SoundLevelType::InputEq),
        soundLevels.at(SoundLevelType::OutputGain)));
}
