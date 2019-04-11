#include <Communication/Messages/Initialization/RelaunchInitializationMessage.h>

using namespace adaptone;

constexpr size_t RelaunchInitializationMessage::SeqId;

RelaunchInitializationMessage::RelaunchInitializationMessage() : ApplicationMessage(SeqId)
{
}

RelaunchInitializationMessage::~RelaunchInitializationMessage()
{
}
