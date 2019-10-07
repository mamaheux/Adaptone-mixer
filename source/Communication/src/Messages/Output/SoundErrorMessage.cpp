#include <Communication/Messages/Output/SoundErrorMessage.h>

using namespace adaptone;
using namespace std;

SoundErrorPosition::SoundErrorPosition() :
    m_x(0),
    m_y(0),
    m_type(PositionType::Speaker),
    m_errorRate(0)
{
}

SoundErrorPosition::SoundErrorPosition(double x, double y, PositionType type, double errorRate) :
    m_x(x),
    m_y(y),
    m_type(type),
    m_errorRate(errorRate)
{
}

SoundErrorPosition::~SoundErrorPosition()
{
}

constexpr size_t SoundErrorMessage::SeqId;

SoundErrorMessage::SoundErrorMessage() : ApplicationMessage(SeqId)
{
}

SoundErrorMessage::SoundErrorMessage(const vector<SoundErrorPosition>& positions) :
    ApplicationMessage(SeqId),
    m_positions(positions)
{
}

SoundErrorMessage::~SoundErrorMessage()
{
}

