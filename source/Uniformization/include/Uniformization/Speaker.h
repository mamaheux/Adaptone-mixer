#ifndef UNIFORMIZATION_SPEAKER_H
#define UNIFORMIZATION_SPEAKER_H

namespace adaptone
{
    class Speaker
    {
        float m_coord[3];

    public:
        Speaker();
        Speaker(float x, float y);
        Speaker(float x, float y, float z);
        ~Speaker();

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
