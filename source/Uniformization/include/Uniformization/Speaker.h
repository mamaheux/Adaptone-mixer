#ifndef UNIFORMIZATION_SPEAKER_H
#define UNIFORMIZATION_SPEAKER_H

namespace adaptone
{
    class Speaker
    {
        float m_coord[2];

    public:
        Speaker();
        Speaker(float x, float y);
        ~Speaker();

        float* getSpeakerCoord();
        float getSpeakerX();
        float getSpeakerY();
    };

    float* Speaker::getSpeakerCoord()
    {
        return m_coord;
    }

    float Speaker::getSpeakerX()
    {
        return m_coord[0];
    }

    float Speaker::getSpeakerY()
    {
        return m_coord[1];
    }
}
#endif
