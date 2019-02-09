#if defined(__unix__) || defined(__linux__)

#include <Mixer/Audio/Alsa/AlsaException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(AlsaExceptionTests, construtor_shouldSetTheRightMessage)
{
    try
    {
        THROW_ALSA_EXCEPTION("my_message", -1, "my_description");
    }
    catch (exception& ex)
    {
        string message = ex.what();
        EXPECT_NE(message.find("AlsaExceptionTests.cpp"), string::npos);
        EXPECT_NE(message.find("construtor_shouldSetTheRightMessage"), string::npos);
        EXPECT_NE(message.find("14"), string::npos);
        EXPECT_NE(message.find("AlsaException:"), string::npos);
        EXPECT_NE(message.find("my_message"), string::npos);
        EXPECT_NE(message.find("-1"), string::npos);
        EXPECT_NE(message.find("my_description"), string::npos);
    }
}

#endif
