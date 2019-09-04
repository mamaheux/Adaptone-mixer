#include <Mixer/MixerAnalysisDispatcher.h>

#include <Utils/Exception/InvalidValueException.h>
#include <Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;
using ::testing::HasSubstr;

class LoggerMock : public Logger
{
public:
    LoggerMock()
    {}

    ~LoggerMock() override
    {}

    MOCK_METHOD1(logMessage, void(const string&));
};

map<AnalysisDispatcher::SoundLevelType, vector<float>> getDummySoundLevels()
{
    map<AnalysisDispatcher::SoundLevelType, vector<float>> soundLevels;
    soundLevels[AnalysisDispatcher::SoundLevelType::InputGain] = {1};
    soundLevels[AnalysisDispatcher::SoundLevelType::InputEq] = {3};
    soundLevels[AnalysisDispatcher::SoundLevelType::OutputGain] = {5};
    return soundLevels;
}

constexpr std::size_t FrameSampleCount = 10;
constexpr std::size_t SampleFrequency = 48000;
constexpr std::size_t InputChannelCount = 2;

TEST(MixerAnalysisDispatcherTests, startStop_shouldStartAndStopProperlyTheDispatcher)
{
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    MixerAnalysisDispatcher dispatcher(logger,
        [](const ApplicationMessage& m) { },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount);

    dispatcher.start();
    this_thread::sleep_for(100ms);
    dispatcher.stop();
}

TEST(MixerAnalysisDispatcherTests, thread_shouldLogExceptions)
{
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    EXPECT_CALL((*logger.get()), logMessage(HasSubstr("InvalidValueException")));


    MixerAnalysisDispatcher dispatcher(logger,
        [](const ApplicationMessage& m)
        {
            THROW_INVALID_VALUE_EXCEPTION("", "");
        },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount);

    dispatcher.start();
    dispatcher.notifySoundLevel(getDummySoundLevels());
    this_thread::sleep_for(100ms);
    dispatcher.stop();
}

TEST(MixerAnalysisDispatcherTests, thread_shouldSendMessages)
{
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();

    map<AnalysisDispatcher::SoundLevelType, vector<float>> soundLevels = getDummySoundLevels();

    MixerAnalysisDispatcher dispatcher(logger,
        [&](const ApplicationMessage& m)
        {
            const SoundLevelMessage* message = dynamic_cast<const SoundLevelMessage*>(&m);
            if (message != nullptr)
            {
                EXPECT_EQ(message->inputAfterGain()[0], soundLevels[AnalysisDispatcher::SoundLevelType::InputGain][0]);
                EXPECT_EQ(message->inputAfterEq()[0], soundLevels[AnalysisDispatcher::SoundLevelType::InputEq][0]);
                EXPECT_EQ(message->outputAfterGain()[0], soundLevels[AnalysisDispatcher::SoundLevelType::OutputGain][0]);
            }
            else
            {
                FAIL();
            }
        },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount);

    dispatcher.start();
    dispatcher.notifySoundLevel(soundLevels);
    this_thread::sleep_for(100ms);
    dispatcher.stop();
}

TEST(MixerAnalysisDispatcherTests, notifyInputEqOutputFrame_wrongProcessingType_shouldThrowNotSupportedException)
{
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    MixerAnalysisDispatcher floatDispatcher(logger,
        [](const ApplicationMessage& m) { },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount);

    MixerAnalysisDispatcher doubleDispatcher(logger,
        [](const ApplicationMessage& m) { },
        ProcessingDataType::Double,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount);


    floatDispatcher.start();
    doubleDispatcher.start();

    EXPECT_THROW(floatDispatcher.notifyInputEqOutputFrame([](double* b) { }), NotSupportedException);
    EXPECT_THROW(doubleDispatcher.notifyInputEqOutputFrame([](float* b) { }), NotSupportedException);

    this_thread::sleep_for(100ms);
    floatDispatcher.stop();
    doubleDispatcher.stop();
}
