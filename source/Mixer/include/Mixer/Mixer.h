#ifndef MIXER_MIXER_H
#define MIXER_MIXER_H

#include <Mixer/Configuration/Configuration.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <memory>

namespace adaptone
{
    class Mixer
    {
        Configuration m_configuration;
        std::shared_ptr<Logger> m_logger;

    public:
        Mixer(const Configuration& configuration);
        virtual ~Mixer();

        int run();

    private:
        std::shared_ptr<Logger> createLogger();
    };
}

#endif