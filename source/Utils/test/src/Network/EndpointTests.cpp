#include <Utils/Network/Endpoint.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(EndpointTests, defaultConstructor_shouldSetTheAttributes)
{
    Endpoint endpoint;

    EXPECT_EQ(endpoint.ipAddress(), "");
    EXPECT_EQ(endpoint.port(), 0);
}

TEST(EndpointTests, constructor_shouldSetTheAttributes)
{
    constexpr const char* IpAddress = "192.168.1.2";
    constexpr uint16_t Port = 1000;
    Endpoint endpoint(IpAddress, Port);

    EXPECT_EQ(endpoint.ipAddress(), IpAddress);
    EXPECT_EQ(endpoint.port(), Port);
}

TEST(EndpointTests, valueParser_ipv4NoColon_shouldThrowPropertyParseException)
{
    EXPECT_THROW(ValueParser<Endpoint>::parse("key", "192.168.1.1"), PropertyParseException);
}

TEST(EndpointTests, valueParser_ipv4NoPort_shouldThrowPropertyParseException)
{
    EXPECT_THROW(ValueParser<Endpoint>::parse("key", "192.168.1.1:"), PropertyParseException);
}

TEST(EndpointTests, valueParser_ipv4InvalidPort_shouldThrowPropertyParseException)
{
    EXPECT_THROW(ValueParser<Endpoint>::parse("key", "192.168.1.1:a"), PropertyParseException);
}

TEST(EndpointTests, valueParser_ipv4_shouldReturnTheSpecifiedEndpoint)
{
    Endpoint endpoint = ValueParser<Endpoint>::parse("key", "192.168.1.1:1000");

    EXPECT_EQ(endpoint.ipAddress(), "192.168.1.1");
    EXPECT_EQ(endpoint.port(), 1000);
}


TEST(EndpointTests, valueParser_ipv6NoColon_shouldThrowPropertyParseException)
{
    EXPECT_THROW(ValueParser<Endpoint>::parse("key", "[2002::]"), PropertyParseException);
}

TEST(EndpointTests, valueParser_ipv6NoPort_shouldThrowPropertyParseException)
{
    EXPECT_THROW(ValueParser<Endpoint>::parse("key", "[2002::]:"), PropertyParseException);
}

TEST(EndpointTests, valueParser_ipv6InvalidPort_shouldThrowPropertyParseException)
{
    EXPECT_THROW(ValueParser<Endpoint>::parse("key", "[2002::]:a"), PropertyParseException);
}

TEST(EndpointTests, valueParser_ipv6NoLastBracket_shouldThrowPropertyParseException)
{
    EXPECT_THROW(ValueParser<Endpoint>::parse("key", "[2002::5:1000"), PropertyParseException);
}

TEST(EndpointTests, valueParser_ipv6_shouldReturnTheSpecifiedEndpoint)
{
    Endpoint endpoint = ValueParser<Endpoint>::parse("key", "[2002::]:1000");

    EXPECT_EQ(endpoint.ipAddress(), "2002::");
    EXPECT_EQ(endpoint.port(), 1000);
}
