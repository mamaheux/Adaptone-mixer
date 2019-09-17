#ifndef MIXER_CONFIGURATION_UNIFORMIZATION_CONFIGURATION_H
#define MIXER_CONFIGURATION_UNIFORMIZATION_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>

namespace adaptone
{
    class UniformizationConfiguration
    {
    public:
        explicit UniformizationConfiguration(const Properties& properties);
        virtual ~UniformizationConfiguration();
    };
}

#endif
