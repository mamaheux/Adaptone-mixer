#include <Communication/Messages/Input/ToogleUniformizationMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t ToogleUniformizationMessage::SeqId;

ToogleUniformizationMessage::ToogleUniformizationMessage() : ApplicationMessage(SeqId), m_isOn(false)
{
}

ToogleUniformizationMessage::ToogleUniformizationMessage(bool isOn) : ApplicationMessage(SeqId), m_isOn(isOn)
{
}

ToogleUniformizationMessage::~ToogleUniformizationMessage()
{
}
