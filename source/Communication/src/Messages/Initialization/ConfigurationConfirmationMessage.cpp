#include <Communication/Messages/Initialization/ConfigurationConfirmationMessage.h>

using namespace adaptone;

constexpr size_t ConfigurationConfirmationMessage::SeqId;

ConfigurationConfirmationMessage::ConfigurationConfirmationMessage() : ApplicationMessage(SeqId)
{
}

ConfigurationConfirmationMessage::~ConfigurationConfirmationMessage()
{
}
