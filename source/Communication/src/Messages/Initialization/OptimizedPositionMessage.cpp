#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t OptimizedPositionMessage::SeqId;

OptimizedPositionMessage::OptimizedPositionMessage() : ApplicationMessage(SeqId),
    m_positions()
{
}

OptimizedPositionMessage::OptimizedPositionMessage(const vector<ConfigurationPosition>& positions) :
    ApplicationMessage(SeqId),
    m_positions(positions)
{
}

OptimizedPositionMessage::~OptimizedPositionMessage()
{
}
