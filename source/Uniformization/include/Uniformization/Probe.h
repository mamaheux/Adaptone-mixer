#ifndef UNIFORMIZATION_PROBE_H
#define UNIFORMIZATION_PROBE_H

namespace adaptone
{
    class Probe
    {
        float m_coord[2];

    public:
        Probe();
        Probe(float x, float y);
        ~Probe();

        float* getProbeCoord();
        float getProbeX();
        float getProbeY();
    };

    float* Probe::getProbeCoord()
    {
        return m_coord;
    }

    float Probe::getProbeX()
    {
        return m_coord[0];
    }

    float Probe::getProbeY()
    {
        return m_coord[1];
    }
}
#endif
