#include <Mixer/Configuration/WebSocketConfiguration.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(WebSocketConfigurationTests, constructor_shouldSetTheAttributes)
{
    WebSocketConfiguration configuration(Properties(
    {
        { "web_socket.endpoint", "^/echo/?$" },
        { "web_socket.port", "8080" }
    }));

    EXPECT_EQ(configuration.endpoint(), "^/echo/?$");
    EXPECT_EQ(configuration.port(), 8080);
}
