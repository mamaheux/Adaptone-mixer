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
        Room(std::size_t speakerCount, std::size_t probeCount);
        Room(const arma::mat& speakerPosMat, const arma::mat& probePosMat);
        virtual ~Room();

        const std::vector<Probe>& probes() const;
        const std::vector<Speaker>& speakers() const;

        void setProbesId(const std::vector<uint32_t>& ids);
        void setSpeakersId(const std::vector<uint32_t>& ids);
        void setSpeakerDirectivities(std::size_t speakerIndex, const arma::vec& directivities);

        void setProbesPosFromMat(const arma::mat posMat);
        void setSpeakersPosFromMat(const arma::mat posMat);
        arma::mat getProbesPosMat() const;
        arma::mat getSpeakersPosMat() const;
    };

    inline const std::vector<Probe>& Room::probes() const
    {
        return m_probes;
    }

    inline const std::vector<Speaker>& Room::speakers() const
    {
        return m_speakers;
    }
}

#endif
