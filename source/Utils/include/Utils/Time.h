#ifndef UTILS_TIME_H
#define UTILS_TIME_H

#include <time.h>
#include <cmath>

namespace adaptone
{
    inline void waitUntilTimeReached(const timespec& time)
    {
        timespec ts;
        do
        {
            timespec_get(&ts, TIME_UTC);
        } while (ts.tv_sec < time.tv_sec || (ts.tv_sec == time.tv_sec && ts.tv_nsec < time.tv_nsec));
    }

    inline timespec addMsToTimespec(size_t delayMs, const timespec& time)
    {
        constexpr size_t SecToMs = 1000;
        constexpr size_t MsToNs = 1000000;
        constexpr size_t SecToNs = 1000000000;

        timespec ts = time;
        ts.tv_sec += delayMs / SecToMs;
        ts.tv_nsec += (delayMs % SecToMs) * MsToNs;
        if (ts.tv_nsec >= SecToNs)
        {
            ts.tv_sec += ts.tv_nsec / SecToNs;
            ts.tv_nsec = ts.tv_nsec % SecToNs;
        }

        return ts;
    }
}

#endif
