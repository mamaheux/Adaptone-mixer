#ifndef UNIFORMIZATION_PROBE_H
#define UNIFORMIZATION_PROBE_H

namespace adaptone
{
    class Probe
    {
        float m_coord[3];

    public:
        Probe();
        Probe(float x, float y);
        Probe(float x, float y, float z);
        ~Probe();

        void setCoord(const float coord[3]);
        void setX(float x);
        void setY(float y);
        void setZ(float z);
        float* getCoord();
        float getX();
        float getY();
        float getZ();
    };
}
#endif
