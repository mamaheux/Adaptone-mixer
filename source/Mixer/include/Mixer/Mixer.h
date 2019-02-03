#ifndef MIXER_MIXER_H
#define MIXER_MIXER_H

#include <Mixer/Configuration/Configuration.h>
#include <Mixer/AudioInput/AudioInput.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <memory>

namespace adaptone
{
    class Mixer
    {
        Configuration m_configuration;
        std::shared_ptr<Logger> m_logger;
        std::unique_ptr<AudioInput> m_audioInput;

    public:
        Mixer(const Configuration& configuration);
        virtual ~Mixer();

        int run();

    private:
        std::shared_ptr<Logger> createLogger();
        std::unique_ptr<AudioInput> createAudioInput();
    };
}

#endif