#ifndef UNIFORMIZATION_MODEL_SPEAKER_H
#define UNIFORMIZATION_MODEL_SPEAKER_H

namespace adaptone
{
    class Speaker
    {
        float m_x;
        float m_y;
        float m_z;

    public:
        Speaker();
        Speaker(float x, float y);
        Speaker(float x, float y, float z);
        ~Speaker();

        void setX(float x);
        void setY(float y);
        void setZ(float z);
        float getX() const;
        float getY() const;
        float getZ() const;
    };

    inline void Speaker::setX(float x)
    {
        m_x = x;
    }

    inline void Speaker::setY(float y)
    {
        m_y = y;
    }

    inline void Speaker::setZ(float z)
    {
        m_z = z;
    }

    inline float Speaker::getX() const
    {
        return m_x;
    }

    inline float Speaker::getY() const
    {
        return m_y;
    }

    inline float Speaker::getZ() const
    {
        return m_z;
    }
}

#endif
