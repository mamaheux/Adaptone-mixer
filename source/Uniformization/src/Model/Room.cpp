#include <Uniformization/Model/Room.h>

using namespace adaptone;

Room::Room(size_t speakerCount, size_t probeCount) : m_speakerCount(speakerCount), m_probeCount(probeCount)
{
    for (int i = 0; i < m_speakerCount; i++)
    {
        m_speakers.push_back(Speaker(i,0,0));
    }

    for (int i = 0; i < m_probeCount; i++)
    {
        m_probes.push_back(Probe(i,5,0));
    }
}

Room::Room(const arma::mat& speakerPosMat, const arma::mat& probePosMat) :
    Room(speakerPosMat.n_rows, probePosMat.n_rows)
{
    setSpeakersPosFromMat(speakerPosMat);
    setProbesPosFromMat(probePosMat);
}

Room::~Room()
{
}

void Room::setProbesId(const std::vector<size_t>& ids)
{
    for (int i = 0; i < m_probes.size(); i++)
    {
        m_probes[i].setId(ids[i]);
    }
}

void Room::setSpeakersId(const std::vector<size_t>& ids)
{
    for (int i = 0; i < m_speakers.size(); i++)
    {
        m_speakers[i].setId(ids[i]);
    }
}

void Room::setSpeakerDirectivities(size_t speakerIndex, const arma::vec& directivities)
{
    m_speakers[speakerIndex].setDirectivities(directivities);
}

void Room::setProbesPosFromMat(const arma::mat posMat)
{
    if (posMat.n_rows != m_probeCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("posMat.n_rows != m_probeCount", "");
    }
    for (int i = 0; i < m_probeCount; i++)
    {
        m_probes[i].setX(posMat(i,0));
        m_probes[i].setY(posMat(i,1));
        m_probes[i].setZ(posMat(i,2));
    }
}

void Room::setSpeakersPosFromMat(const arma::mat posMat)
{
    if (posMat.n_rows != m_speakerCount)
    {
        THROW_INVALID_VALUE_EXCEPTION("posMat.n_rows != m_probeCount", "");
    }
    for (int i = 0; i < m_speakerCount; i++)
    {
        m_speakers[i].setX(posMat(i,0));
        m_speakers[i].setY(posMat(i,1));
        m_speakers[i].setZ(posMat(i,2));
    }
}

arma::mat Room::getProbesPosMat() const
{
    arma::mat probesPosMat = arma::zeros(m_probeCount, 3);
    for (int i = 0; i < m_probeCount; i++)
    {
        probesPosMat(i,0) = m_probes[i].x();
        probesPosMat(i,1) = m_probes[i].y();
        probesPosMat(i,2) = m_probes[i].z();
    }
    return probesPosMat;
}

arma::mat Room::getSpeakersPosMat() const
{
    arma::mat speakersPosMat = arma::zeros(m_speakerCount, 3);
    for (int i = 0; i < m_speakerCount; i++)
    {
        speakersPosMat(i,0) = m_speakers[i].x();
        speakersPosMat(i,1) = m_speakers[i].y();
        speakersPosMat(i,2) = m_speakers[i].z();
    }
    return speakersPosMat;
}
