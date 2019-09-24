#include <Utils/Exception/NetworkException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(NetworkExceptionTests, constructor_shouldSetTheRightMessage)
{
    try
    {
        THROW_NETWORK_EXCEPTION("my message");
    }
    catch (exception& ex)
    {
        string message = ex.what();
        EXPECT_NE(message.find("NetworkExceptionTests.cpp"), string::npos);
        EXPECT_NE(message.find("constructor_shouldSetTheRightMessage"), string::npos);
        EXPECT_NE(message.find("12"), string::npos);
        EXPECT_NE(message.find("NetworkException:"), string::npos);
        EXPECT_NE(message.find("my message"), string::npos);
    }
}
