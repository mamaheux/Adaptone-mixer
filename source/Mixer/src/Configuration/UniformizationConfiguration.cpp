#include <Mixer/Configuration/UniformizationConfiguration.h>

using namespace adaptone;

UniformizationConfiguration::UniformizationConfiguration(const Properties& properties)
{
    constexpr const char* RoutineIRSweepF1PropertyKey = "uniformization.routineIRsweepF1";
    constexpr const char* RoutineIRSweepF2PropertyKey = "uniformization.routineIRsweepF1";
    constexpr const char* RoutineIRSweepTPropertyKey = "uniformization.routineIRsweepT";

    m_routineIRSweepF1 = properties.get<float>(RoutineIRSweepF1PropertyKey);
    m_routineIRSweepF2 = properties.get<float>(RoutineIRSweepF2PropertyKey);
    m_routineIRSweepT = properties.get<float>(RoutineIRSweepTPropertyKey);
}

UniformizationConfiguration::~UniformizationConfiguration()
{
}
