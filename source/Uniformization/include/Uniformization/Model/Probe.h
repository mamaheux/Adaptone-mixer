#ifndef UNIFORMIZATION_MODEL_PROBE_H
#define UNIFORMIZATION_MODEL_PROBE_H

namespace adaptone
{
    class Probe
    {
        float m_x;
        float m_y;
        float m_z;

    public:
        Probe();
        Probe(float x, float y);
        Probe(float x, float y, float z);
        ~Probe();

        void setX(float x);
        void setY(float y);
        void setZ(float z);
        float getX() const;
        float getY() const;
        float getZ() const;
    };

    inline void Probe::setX(float x)
    {
        m_x = x;
    }

    inline void Probe::setY(float y)
    {
        m_y = y;
    }

    inline void Probe::setZ(float z)
    {
        m_z = z;
    }

    inline float Probe::getX() const
    {
        return m_x;
    }

    inline float Probe::getY() const
    {
        return m_y;
    }

    inline float Probe::getZ() const
    {
        return m_z;
    }
}

#endif
