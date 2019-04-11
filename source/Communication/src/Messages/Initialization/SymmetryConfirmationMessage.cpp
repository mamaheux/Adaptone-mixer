#include <Communication/Messages/Initialization/SymmetryConfirmationMessage.h>

using namespace adaptone;
using namespace std;

constexpr size_t SymmetryConfirmationMessage::SeqId;

SymmetryConfirmationMessage::SymmetryConfirmationMessage() : ApplicationMessage(SeqId),
    m_symmetry(0)
{
}

SymmetryConfirmationMessage::SymmetryConfirmationMessage(size_t symmetry) : ApplicationMessage(SeqId),
    m_symmetry(symmetry)
{
}

SymmetryConfirmationMessage::~SymmetryConfirmationMessage()
{
}
