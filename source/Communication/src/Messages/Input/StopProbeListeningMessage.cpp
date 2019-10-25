#include <Communication/Messages/Input/StopProbeListeningMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t StopProbeListeningMessage::SeqId;

StopProbeListeningMessage::StopProbeListeningMessage() : ApplicationMessage(SeqId)
{
}

StopProbeListeningMessage::~StopProbeListeningMessage()
{
}
