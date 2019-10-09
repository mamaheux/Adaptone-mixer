#include <Uniformization/Model/Room.h>

using namespace adaptone;

Room::Room()
{
    m_speakerCount = 0;
    m_probeCount = 0;
}

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

void Room::setProbesPosFromMat(const arma::mat posMat)
{
    if (posMat.n_rows != m_probeCount)
    {
        //TODO
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
        //TODO
    }
    for (int i = 0; i < m_speakerCount; i++)
    {
        m_speakers[i].setX(posMat(i,0));
        m_speakers[i].setY(posMat(i,1));
        m_speakers[i].setZ(posMat(i,2));
    }
}

arma::mat Room::getProbesPosMat()
{
    arma::mat probesPosMat = arma::zeros(m_probeCount, 3);
    for (int i = 0; i < m_probeCount; i++)
    {
        probesPosMat(i,0) = m_probes[i].getX();
        probesPosMat(i,1) = m_probes[i].getY();
        probesPosMat(i,2) = m_probes[i].getZ();
    }
    return probesPosMat;
}

arma::mat Room::getSpeakersPosMat()
{
    arma::mat speakersPosMat = arma::zeros(m_speakerCount, 3);
    for (int i = 0; i < m_speakerCount; i++)
    {
        speakersPosMat(i,0) = m_speakers[i].getX();
        speakersPosMat(i,1) = m_speakers[i].getY();
        speakersPosMat(i,2) = m_speakers[i].getZ();
    }
    return speakersPosMat;
}
