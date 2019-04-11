#include <Communication/Messages/Initialization/OptimizePositionMessage.h>

using namespace adaptone;

constexpr size_t OptimizePositionMessage::SeqId;

OptimizePositionMessage::OptimizePositionMessage() : ApplicationMessage(SeqId)
{
}

OptimizePositionMessage::~OptimizePositionMessage()
{
}
