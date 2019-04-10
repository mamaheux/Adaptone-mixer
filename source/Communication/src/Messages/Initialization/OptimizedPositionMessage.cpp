#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>

using namespace adaptone;
using namespace std;

OptimizedPositionMessage::OptimizedPositionMessage() : ApplicationMessage(7),
    m_positions()
{
}

OptimizedPositionMessage::OptimizedPositionMessage(const vector<ConfigurationPosition>& positions) :
    ApplicationMessage(7),
    m_positions(positions)
{
}

OptimizedPositionMessage::~OptimizedPositionMessage()
{
}
