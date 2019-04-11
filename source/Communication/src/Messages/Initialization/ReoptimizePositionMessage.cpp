#include <Communication/Messages/Initialization/ReoptimizePositionMessage.h>

using namespace adaptone;

constexpr size_t ReoptimizePositionMessage::SeqId;

ReoptimizePositionMessage::ReoptimizePositionMessage() : ApplicationMessage(SeqId)
{
}

ReoptimizePositionMessage::~ReoptimizePositionMessage()
{
}
