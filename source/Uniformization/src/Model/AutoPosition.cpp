#include <Uniformization/Model/AutoPosition.h>

#include <armadillo>

#include <cmath>

using namespace arma;
using namespace adaptone;
using namespace std;

AutoPosition::AutoPosition(double alpha,
    double epsilonTotalDistanceError,
    double epsilonDeltaTotalDistanceError,
    size_t iterationCount,
    size_t thermalIterationCount,
    size_t tryCount,
    size_t countThreshold) :
    m_alpha(alpha),
    m_epsilonTotalDistanceError(epsilonTotalDistanceError),
    m_epsilonDeltaTotalDistanceError(epsilonDeltaTotalDistanceError),
    m_iterationCount(iterationCount),
    m_thermalIterationCount(thermalIterationCount),
    m_tryCount(tryCount),
    m_countThreshold(countThreshold)
{
}

AutoPosition::~AutoPosition()
{
}

void AutoPosition::computeRoomConfiguration2D(Room& room, const mat& distancesMat, float distanceRelativeError,
    bool randomInitConfiguration)
{
    //===================================================
    // 1) Get relative positions from distances matrix
    //===================================================

    mat speakersPosMat = room.getSpeakersPosMat();
    mat probesPosMat = room.getProbesPosMat();

    double avgDist = arma::mean(arma::mean(distancesMat));

    // Force the room to have no initial configuration if one is already defined
    if (randomInitConfiguration)
    {
        speakersPosMat.clear();
        probesPosMat.clear();

        speakersPosMat = avgDist * randu<mat>(distancesMat.n_rows, 2);
        probesPosMat = avgDist * randu<mat>(distancesMat.n_cols, 2);
        // Space out the probes from the speakers for better convergence
        probesPosMat.col(1) += 2*avgDist;
    }

    // Compute the epsilonTotalDistError value according to distances predicted relative error if one is defined
    double epsilonTotalDistError;
    if (distanceRelativeError > 0.0)
    {
        epsilonTotalDistError = avgDist * distanceRelativeError;
    }
    else
    {
        epsilonTotalDistError = m_epsilonTotalDistanceError;
    }

    // Compute relatives positions from distances matrix
    computeRelativePositionsFromDistances(distancesMat, m_iterationCount, m_tryCount, m_thermalIterationCount, m_alpha,
        epsilonTotalDistError, m_epsilonDeltaTotalDistanceError,m_countThreshold, 2, speakersPosMat, probesPosMat);

    // Resize speakers/probesPosMat to be in 3 dimension to match room data format
    speakersPosMat.resize(speakersPosMat.n_rows,3);
    speakersPosMat.col(2) = zeros(speakersPosMat.n_rows,1);
    probesPosMat.resize(probesPosMat.n_rows,3);
    probesPosMat.col(2) = zeros(probesPosMat.n_rows,1);

    //===================================================
    // 2) Apply proper rotation to relative positions
    //===================================================

    // Make all positions origin based on Speakers centroid
    vec speakersCentroid = getSetCentroid(speakersPosMat);
    vec probesCentroid = getSetCentroid(probesPosMat);
    moveSet(speakersPosMat, -speakersCentroid);
    moveSet(probesPosMat, -speakersCentroid);

    // Determined the proper angle to apply
    float angleS = findSetAngle2D(speakersPosMat); //return value between -pi/2 and pi/2
    float angleSP = atan2(probesCentroid(1) - speakersCentroid(1), probesCentroid(0) - speakersCentroid(1));
    int probesParity = sign(angleSP);//sign(probesCentroid(0) - speakersCentroid(1));
    int speakerParity = sign(angleS);

    float rotAngle;
    if (probesParity > 0)
    {
        rotAngle = -angleS;
    }
    else
    {
        rotAngle = speakerParity * M_PI - angleS;
    }

    // Apply rotation
    rotateSet2D(speakersPosMat, rotAngle);
    rotateSet2D(probesPosMat, rotAngle);

    //===================================================
    // 3) Update Room with new Probe and Speaker positions
    //===================================================

    room.setProbesPosFromMat(probesPosMat);
    room.setSpeakersPosFromMat(speakersPosMat);
}
