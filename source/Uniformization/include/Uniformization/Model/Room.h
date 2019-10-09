#ifndef UNIFORMIZATION_MODEL_ROOM_H
#define UNIFORMIZATION_MODEL_ROOM_H

#include <Uniformization/Model/Probe.h>
#include <Uniformization/Model/Speaker.h>

#include <Utils/Exception/InvalidValueException.h>

#include <vector>
#include <memory>
#include <armadillo>

namespace adaptone
{
    class Room
    {
        std::size_t m_probeCount;
        std::size_t m_speakerCount;
        std::vector<Probe> m_probes;
        std::vector<Speaker> m_speakers;

    public:
        Room();
        Room(std::size_t speakerCount, std::size_t probeCount);
        Room(const arma::mat& speakerPosMat, const arma::mat& probePosMat);
        ~Room();

        void setProbesPosFromMat(const arma::mat posMat);
        void setSpeakersPosFromMat(const arma::mat posMat);
        arma::mat getProbesPosMat();
        arma::mat getSpeakersPosMat();
    };
}

#endif
