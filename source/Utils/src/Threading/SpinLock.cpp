#include <Utils/Threading/SpinLock.h>

using namespace adaptone;
using namespace std;

SpinLock::SpinLock()
{
    m_flag.clear(memory_order_release);
}

SpinLock::~SpinLock()
{
}
