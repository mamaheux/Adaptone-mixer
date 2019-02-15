#ifndef MIXER_MIXER_H
#define MIXER_MIXER_H

#include <Mixer/Configuration/Configuration.h>
#include <Mixer/AudioInput/AudioInput.h>
#include <Mixer/AudioOutput/AudioOutput.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <memory>
#include <atomic>

namespace adaptone
{
    class Mixer
    {
        Configuration m_configuration;
        std::shared_ptr<Logger> m_logger;

        std::unique_ptr<AudioInput> m_audioInput;
        std::unique_ptr<AudioOutput> m_audioOutput;

        std::atomic<bool> m_stopped;

    public:
        Mixer(const Configuration& configuration);
        virtual ~Mixer();

        int run();
        void stop();

    private:
        std::shared_ptr<Logger> createLogger();

        std::unique_ptr<AudioInput> createAudioInput();
        std::unique_ptr<AudioOutput> createAudioOutput();
    };
}

#endif
