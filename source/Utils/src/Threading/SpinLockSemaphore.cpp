#include <Utils/Threading/SpinLockSemaphore.h>

using namespace adaptone;
SpinLockSemaphore::SpinLockSemaphore(int count) : m_count(count)
{
    m_waitSpinLock.lock();
}

SpinLockSemaphore::~SpinLockSemaphore()
{
}
