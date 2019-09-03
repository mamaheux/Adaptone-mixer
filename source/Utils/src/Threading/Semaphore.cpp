#include <Utils/Threading/Semaphore.h>

using namespace adaptone;
using namespace std;

Semaphore::Semaphore(size_t count) : m_count(count)
{
}

Semaphore::~Semaphore()
{
}
