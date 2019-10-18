#include <Uniformization/Communication/ProbeDiscoverer.h>

#include <gtest/gtest.h>

#include <chrono>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;

TEST(ProbeDiscovererTests, discover_shouldTryTheSpecifiedTrialCount)
{
    constexpr double MaxAbsElapsedMsTimeError = 2500;
    constexpr int TimeoutMs = 100;
    constexpr size_t DiscoveryTrialCount = 10;

    ProbeDiscoverer probeDiscoverer(Endpoint("192.168.0.255", 5000), TimeoutMs, DiscoveryTrialCount);

    auto start = chrono::system_clock::now();
    probeDiscoverer.discover();
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsedSeconds = end - start;

    EXPECT_NEAR(elapsedSeconds.count() * 1000, TimeoutMs * DiscoveryTrialCount, MaxAbsElapsedMsTimeError);
}
