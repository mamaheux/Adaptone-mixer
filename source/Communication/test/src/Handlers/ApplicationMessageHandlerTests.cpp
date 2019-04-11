#include <Communication/Handlers/ApplicationMessageHandler.h>

#include <Communication/Messages/Initialization/ConfigurationChoiceMessage.h>
#include <Communication/Messages/Initialization/InitialParametersCreationMessage.h>
#include <Communication/Messages/Initialization/LaunchInitializationMessage.h>
#include <Communication/Messages/Initialization/PositionConfirmationMessage.h>
#include <Communication/Messages/Initialization/RelaunchInitializationMessage.h>
#include <Communication/Messages/Initialization/SymmetryConfirmationMessage.h>
#include <Communication/Messages/Initialization/OptimizePositionMessage.h>
#include <Communication/Messages/Initialization/OptimizedPositionMessage.h>
#include <Communication/Messages/Initialization/ReoptimizePositionMessage.h>
#include <Communication/Messages/Initialization/ConfigurationConfirmationMessage.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;

#define DEFINE_TYPE_MATCHER(type) MATCHER(Is##type, "") \
    { \
        return (string(typeid(arg).name()) == typeid(type).name()); \
    }

DEFINE_TYPE_MATCHER(ConfigurationChoiceMessage);
DEFINE_TYPE_MATCHER(InitialParametersCreationMessage);
DEFINE_TYPE_MATCHER(LaunchInitializationMessage);
DEFINE_TYPE_MATCHER(PositionConfirmationMessage);
DEFINE_TYPE_MATCHER(RelaunchInitializationMessage);
DEFINE_TYPE_MATCHER(SymmetryConfirmationMessage);
DEFINE_TYPE_MATCHER(OptimizePositionMessage);
DEFINE_TYPE_MATCHER(OptimizedPositionMessage);
DEFINE_TYPE_MATCHER(ReoptimizePositionMessage);
DEFINE_TYPE_MATCHER(ConfigurationConfirmationMessage);

class ApplicationMessageHandlerMock : public ApplicationMessageHandler
{
public:
    ApplicationMessageHandlerMock()
    {}

    ~ApplicationMessageHandlerMock() override
    {}

    MOCK_METHOD1(handleDeserialized, void(const ApplicationMessage&));
};

TEST(ApplicationMessageHandlerTests, handle_ConfigurationChoiceMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsConfigurationChoiceMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 0,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"monitorsNumber\": 5,"
        "    \"speakersNumber\": 4,"
        "    \"probesNumber\": 8,"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\""
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_InitialParametersCreationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsInitialParametersCreationMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 1,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"monitorsNumber\": 5,"
        "    \"speakersNumber\": 4,"
        "    \"probesNumber\": 8"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_LaunchInitializationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsLaunchInitializationMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 2"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_PositionConfirmationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsPositionConfirmationMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 3,"
        "  \"data\": {"
        "    \"firstSymmetryPositions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\""
        "      }"
        "    ],"
        "    \"secondSymmetryPositions\": ["
        "      {"
        "        \"x\": 340,"
        "        \"y\": 140,"
        "        \"type\": \"s\""
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_RelaunchInitializationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsRelaunchInitializationMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 4"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_SymmetryConfirmationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsSymmetryConfirmationMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 5,"
        "  \"data\": {"
        "    \"symmetry\": 0"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_OptimizePositionMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsOptimizePositionMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 6"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_OptimizedPositionMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsOptimizedPositionMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 7,"
        "  \"data\": {"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\""
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_ReoptimizePositionMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsReoptimizePositionMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 8"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}

TEST(ApplicationMessageHandlerTests, handle_ConfigurationConfirmationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsConfigurationConfirmationMessage()));

    string serializedMessage = "{"
        "  \"seqId\": 9"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j);
}
