#include <Mixer/MixerAnalysisDispatcher.h>

#include <Communication/Messages/Output/InputSpectrumMessage.h>
#include <Communication/Messages/Output/SoundLevelMessage.h>

#include <Utils/Exception/InvalidValueException.h>
#include <Utils/Exception/NotSupportedException.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;
using ::testing::HasSubstr;

constexpr size_t FrameSampleCount = 2;
constexpr size_t SampleFrequency = 48000;
constexpr size_t InputChannelCount = 2;
constexpr size_t SpectrumAnalysisFftLength = 4;
constexpr size_t SpectrumAnalysisPointCountPerDecade = 2;

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

shared_ptr<ChannelIdMapping> getDummyChannelIdMapping()
{
    vector<size_t> inputChannelIds;
    for (size_t i = 0; i < InputChannelCount; i++)
    {
        inputChannelIds.push_back(i);
    }
    shared_ptr<ChannelIdMapping> channelIdMapping =
        make_shared<ChannelIdMapping>(InputChannelCount, 1, vector<size_t>());
    channelIdMapping->update(inputChannelIds, {}, 1);
    return channelIdMapping;
}

TEST(MixerAnalysisDispatcherTests, startStop_shouldStartAndStopProperlyTheDispatcher)
{
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    MixerAnalysisDispatcher dispatcher(logger,
        getDummyChannelIdMapping(),
        [](const ApplicationMessage& m) { },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        SpectrumAnalysisFftLength,
        SpectrumAnalysisPointCountPerDecade);

    dispatcher.start();
    this_thread::sleep_for(100ms);
    dispatcher.stop();
}

TEST(MixerAnalysisDispatcherTests, thread_shouldLogExceptions)
{
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    EXPECT_CALL((*logger.get()), logMessage(HasSubstr("InvalidValueException"))).Times(2);


    MixerAnalysisDispatcher dispatcher(logger,
        getDummyChannelIdMapping(),
        [](const ApplicationMessage& m)
        {
            THROW_INVALID_VALUE_EXCEPTION("", "");
        },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        SpectrumAnalysisFftLength,
        SpectrumAnalysisPointCountPerDecade);

    dispatcher.start();
    dispatcher.notifySoundLevel(getDummySoundLevels());
    this_thread::sleep_for(100ms);
    dispatcher.stop();
}

TEST(MixerAnalysisDispatcherTests, thread_shouldSendMessages)
{
    SoundLevelMessage soundLevelMessage;
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();

    map<AnalysisDispatcher::SoundLevelType, vector<float>> soundLevels = getDummySoundLevels();

    MixerAnalysisDispatcher dispatcher(logger,
        getDummyChannelIdMapping(),
        [&](const ApplicationMessage& m)
        {
            const SoundLevelMessage* message = dynamic_cast<const SoundLevelMessage*>(&m);
            if (message != nullptr)
            {
                soundLevelMessage = *message;
            }
        },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        SpectrumAnalysisFftLength,
        SpectrumAnalysisPointCountPerDecade);

    dispatcher.start();
    dispatcher.notifySoundLevel(soundLevels);
    this_thread::sleep_for(100ms);
    dispatcher.stop();

    EXPECT_EQ(soundLevelMessage.inputAfterGain()[0].channelId(), 0);
    EXPECT_EQ(soundLevelMessage.inputAfterGain()[0].level(), soundLevels[AnalysisDispatcher::SoundLevelType::InputGain][0]);
    EXPECT_EQ(soundLevelMessage.inputAfterEq()[0].channelId(), 0);
    EXPECT_EQ(soundLevelMessage.inputAfterEq()[0].level(), soundLevels[AnalysisDispatcher::SoundLevelType::InputEq][0]);
    EXPECT_EQ(soundLevelMessage.outputAfterGain()[0].channelId(), 0);
    EXPECT_EQ(soundLevelMessage.outputAfterGain()[0].level(), soundLevels[AnalysisDispatcher::SoundLevelType::OutputGain][0]);
}

TEST(MixerAnalysisDispatcherTests, notifyInputEqOutputFrame_wrongProcessingType_shouldThrowNotSupportedException)
{
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    MixerAnalysisDispatcher floatDispatcher(logger,
        getDummyChannelIdMapping(),
        [](const ApplicationMessage& m) { },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        SpectrumAnalysisFftLength,
        SpectrumAnalysisPointCountPerDecade);

    MixerAnalysisDispatcher doubleDispatcher(logger,
        getDummyChannelIdMapping(),
        [](const ApplicationMessage& m) { },
        ProcessingDataType::Double,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        SpectrumAnalysisFftLength,
        SpectrumAnalysisPointCountPerDecade);


    floatDispatcher.start();
    doubleDispatcher.start();

    EXPECT_THROW(floatDispatcher.notifyInputEqOutputFrame([](double* b) { }), NotSupportedException);
    EXPECT_THROW(doubleDispatcher.notifyInputEqOutputFrame([](float* b) { }), NotSupportedException);

    this_thread::sleep_for(100ms);
    floatDispatcher.stop();
    doubleDispatcher.stop();
}

TEST(MixerAnalysisDispatcherTests, notifyInputEqOutputFrame_float_shouldSendInputSpectrumMessage)
{
    bool isReceived = false;
    InputSpectrumMessage inputSpectrumMessage;
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    MixerAnalysisDispatcher floatDispatcher(logger,
        getDummyChannelIdMapping(),
        [&](const ApplicationMessage& m)
        {
            const InputSpectrumMessage* message = dynamic_cast<const InputSpectrumMessage*>(&m);
            if (message != nullptr && !isReceived)
            {
                inputSpectrumMessage = *message;
                isReceived = true;
            }
        },
        ProcessingDataType::Float,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        SpectrumAnalysisFftLength,
        SpectrumAnalysisPointCountPerDecade);


    floatDispatcher.start();

    float frame0[] = { 0, 1, 1, 1 };
    float frame1[] = { 2, 3, 2, 2 };

    floatDispatcher.notifyInputEqOutputFrame([&](float* b)
    {
       memcpy(b, frame0, InputChannelCount * FrameSampleCount * sizeof(float));
    });
    floatDispatcher.notifyInputEqOutputFrame([&](float* b)
    {
       memcpy(b, frame1, InputChannelCount * FrameSampleCount * sizeof(float));
    });

    this_thread::sleep_for(100ms);
    floatDispatcher.stop();

    ASSERT_EQ(inputSpectrumMessage.channelSpectrums().size(), InputChannelCount);

    EXPECT_EQ(inputSpectrumMessage.channelSpectrums()[0].channelId(), 0);
    ASSERT_EQ(inputSpectrumMessage.channelSpectrums()[0].points().size(), 2);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[0].frequency(), 8000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[0].amplitude(), 1.6286497116088867);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[1].frequency(), 16000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[1].amplitude(), 0.52999985218048096);

    EXPECT_EQ(inputSpectrumMessage.channelSpectrums()[1].channelId(), 1);
    ASSERT_EQ(inputSpectrumMessage.channelSpectrums()[1].points().size(), 2);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[0].frequency(), 8000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[0].amplitude(), 1.5823084115982056);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[1].frequency(), 16000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[1].amplitude(), 0.68999993801116943);
}

TEST(MixerAnalysisDispatcherTests, notifyInputEqOutputFrame_double_shouldSendInputSpectrumMessage)
{
    bool isReceived = false;
    InputSpectrumMessage inputSpectrumMessage;
    shared_ptr<LoggerMock> logger = make_shared<LoggerMock>();
    MixerAnalysisDispatcher floatDispatcher(logger,
        getDummyChannelIdMapping(),
        [&](const ApplicationMessage& m)
        {
            const InputSpectrumMessage* message = dynamic_cast<const InputSpectrumMessage*>(&m);
            if (message != nullptr && !isReceived)
            {
                inputSpectrumMessage = *message;
                isReceived = true;
            }
        },
        ProcessingDataType::Double,
        FrameSampleCount,
        SampleFrequency,
        InputChannelCount,
        SpectrumAnalysisFftLength,
        SpectrumAnalysisPointCountPerDecade);


    floatDispatcher.start();

    double frame0[] = { 0, 1, 1, 1 };
    double frame1[] = { 2, 3, 2, 2 };

    floatDispatcher.notifyInputEqOutputFrame([&](double* b)
    {
       memcpy(b, frame0, InputChannelCount * FrameSampleCount * sizeof(double));
    });
    floatDispatcher.notifyInputEqOutputFrame([&](double* b)
    {
       memcpy(b, frame1, InputChannelCount * FrameSampleCount * sizeof(double));
    });

    this_thread::sleep_for(100ms);
    floatDispatcher.stop();

    ASSERT_EQ(inputSpectrumMessage.channelSpectrums().size(), InputChannelCount);

    EXPECT_EQ(inputSpectrumMessage.channelSpectrums()[0].channelId(), 0);
    ASSERT_EQ(inputSpectrumMessage.channelSpectrums()[0].points().size(), 2);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[0].frequency(), 8000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[0].amplitude(), 1.6286497116088867);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[1].frequency(), 16000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[0].points()[1].amplitude(), 0.52999985218048096);

    EXPECT_EQ(inputSpectrumMessage.channelSpectrums()[1].channelId(), 1);
    ASSERT_EQ(inputSpectrumMessage.channelSpectrums()[1].points().size(), 2);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[0].frequency(), 8000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[0].amplitude(), 1.5823084115982056);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[1].frequency(), 16000);
    EXPECT_DOUBLE_EQ(inputSpectrumMessage.channelSpectrums()[1].points()[1].amplitude(), 0.68999993801116943);
}
