#include <Uniformization/Room.h>
#include "Uniformization/Probe.h"
#include "Uniformization/Speaker.h"

using namespace adaptone;

Room::Room()
{
    m_speakerNb = 0;
    m_probeNb = 0;
}

Room::Room(int speakerNb, int probeNb) : m_speakerNb(speakerNb), m_probeNb(probeNb)
{
    for (int i = 0; i < m_speakerNb; i++)
    {
        m_speakers.push_back(Speaker(i,0,0));
    }

    for (int i = 0; i < m_probeNb; i++)
    {
        m_probes.push_back(Probe(i,5,0));
    }

    m_speakersPosMat.resize(m_speakerNb, 3);
    m_probesPosMat.resize(m_probeNb, 3);

    m_speakersPosMat = getSpeakersPosMat();
    m_probesPosMat = getProbesPosMat();
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
    if (posMat.n_rows != m_probeNb)
    {
        //TODO
    }
    for (int i = 0; i < m_probeNb; i++)
    {
        m_probes[i].setX(posMat(i,0));
        m_probes[i].setY(posMat(i,1));
        m_probes[i].setZ(posMat(i,2));
    }
    m_probesPosMat = posMat;
}

void Room::setSpeakersPosFromMat(const arma::mat posMat)
{
    if (posMat.n_rows != m_speakerNb)
    {
        //TODO
    }
    for (int i = 0; i < m_speakerNb; i++)
    {
        m_speakers[i].setX(posMat(i,0));
        m_speakers[i].setY(posMat(i,1));
        m_speakers[i].setZ(posMat(i,2));
    }
    m_speakersPosMat = posMat;
}

arma::mat Room::getProbesPosMat()
{
    for (int i = 0; i < m_probeNb; i++)
    {
        m_probesPosMat(i,0) = m_probes[i].getX();
        m_probesPosMat(i,1) = m_probes[i].getY();
        m_probesPosMat(i,2) = m_probes[i].getZ();
    }
    return m_probesPosMat;
}

arma::mat Room::getSpeakersPosMat()
{
    for (int i = 0; i < m_speakerNb; i++)
    {
        m_speakersPosMat(i,0) = m_speakers[i].getX();
        m_speakersPosMat(i,1) = m_speakers[i].getY();
        m_speakersPosMat(i,2) = m_speakers[i].getZ();
    }
    return m_speakersPosMat;
}
