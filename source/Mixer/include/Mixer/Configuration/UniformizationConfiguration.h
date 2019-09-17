#ifndef MIXER_CONFIGURATION_UNIFORMIZATION_CONFIGURATION_H
#define MIXER_CONFIGURATION_UNIFORMIZATION_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>

namespace adaptone
{
    class UniformizationConfiguration
    {

        float m_routineIRSweepF1;
        float m_routineIRSweepF2;
        float m_routineIRSweepT;

    public:
        explicit UniformizationConfiguration(const Properties& properties);
        virtual ~UniformizationConfiguration();

        float routineIRSweepF1() const;
        float routineIRSweepF2() const;
        float routineIRSweepT() const;
    };


    inline float UniformizationConfiguration::routineIRSweepF1() const
    {
        return m_routineIRSweepF1;
    }

    inline float UniformizationConfiguration::routineIRSweepF2() const
    {
        return m_routineIRSweepF2;
    }

    inline float UniformizationConfiguration::routineIRSweepT() const
    {
        return m_routineIRSweepT;
    }
}





#endif
