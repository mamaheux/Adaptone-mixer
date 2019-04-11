#include <Communication/Messages/Initialization/LaunchInitializationMessage.h>

using namespace adaptone;

constexpr size_t LaunchInitializationMessage::SeqId;

LaunchInitializationMessage::LaunchInitializationMessage() : ApplicationMessage(SeqId)
{
}

LaunchInitializationMessage::~LaunchInitializationMessage()
{
}
