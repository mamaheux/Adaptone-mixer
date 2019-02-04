#include <Utils/Logger/Logger.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stdexcept>

using namespace adaptone;
using namespace std;


class LoggerMock : public Logger
{
public:
    LoggerMock()
    {}

    ~LoggerMock() override
    {}

    MOCK_METHOD1(logMessage, void(
        const string&));
};

TEST(LoggerTests, log_shouldCallTheOverridedMethod)
{
    LoggerMock logger;

    EXPECT_CALL(logger, logMessage(string("Debug --> message 1")));
    EXPECT_CALL(logger, logMessage(string("Information --> exception")));
    EXPECT_CALL(logger, logMessage(string("Warning --> exception --> message 2")));
    EXPECT_CALL(logger, logMessage(string("Error --> message 3")));
    EXPECT_CALL(logger, logMessage(string("Performance --> message 4")));

    logger.log(Logger::Level::Debug, "message 1");
    logger.log(Logger::Level::Information, runtime_error("exception"));
    logger.log(Logger::Level::Warning, runtime_error("exception"), "message 2");
    logger.log(Logger::Level::Error, "message 3");
    logger.log(Logger::Level::Performance, "message 4");
}
