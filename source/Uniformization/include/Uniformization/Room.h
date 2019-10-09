#ifndef UNIFORMIZATION_ROOM_H
#define UNIFORMIZATION_ROOM_H

#include <Uniformization/Probe.h>
#include <Uniformization/Speaker.h>
#include <Utils/Exception/InvalidValueException.h>

#include <vector>
#include <memory>
#include <armadillo>

namespace adaptone
{
    class Room
    {
        int m_probeNb;
        int m_speakerNb;
        std::vector<Probe> m_probes;
        std::vector<Speaker> m_speakers;
        arma::mat m_probesPosMat;
        arma::mat m_speakersPosMat;

    public:
        Room();
        Room(int speakerNb, int probeNb);
        Room(const arma::mat& speakerPosMat, const arma::mat& probePosMat);
        ~Room();

        void setProbesPosFromMat(const arma::mat posMat);
        void setSpeakersPosFromMat(const arma::mat posMat);
        arma::mat getProbesPosMat();
        arma::mat getSpeakersPosMat();
    };
}
#endif
