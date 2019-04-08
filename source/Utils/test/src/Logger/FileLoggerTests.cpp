#include <Utils/Logger/FileLogger.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdio>

using namespace adaptone;
using namespace std;
using ::testing::EndsWith;

constexpr const char* LogFilename = "log.txt";

class FileLoggerTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
        remove(LogFilename);
    }
};

TEST_F(FileLoggerTests, log_shouldWriteInTheFile)
{
    {
        FileLogger logger(LogFilename);
        logger.log(Logger::Level::Warning, "message 1");
        logger.log(Logger::Level::Error, "message 2");
    }
    {
        FileLogger logger(LogFilename);
        logger.log(Logger::Level::Warning, "message 3");
    }

    ifstream logFileStream(LogFilename);

    string lines[3];

    getline(logFileStream, lines[0]);
    getline(logFileStream, lines[1]);
    getline(logFileStream, lines[2]);

    EXPECT_THAT(lines[0], EndsWith("Warning --> message 1"));
    EXPECT_THAT(lines[1], EndsWith("Error --> message 2"));
    EXPECT_THAT(lines[2], EndsWith("Warning --> message 3"));
}
