#include <Communication/Messages/Initialization/SymmetryConfirmationMessage.h>

using namespace adaptone;
using namespace std;

SymmetryConfirmationMessage::SymmetryConfirmationMessage() : ApplicationMessage(5),
    m_symmetry(0)
{
}

SymmetryConfirmationMessage::SymmetryConfirmationMessage(size_t symmetry) : ApplicationMessage(5),
    m_symmetry(symmetry)
{
}

SymmetryConfirmationMessage::~SymmetryConfirmationMessage()
{
}
