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

#include <Communication/Messages/Input/ChangeInputGainMessage.h>
#include <Communication/Messages/Input/ChangeInputGainsMessage.h>
#include <Communication/Messages/Input/ChangeInputEqGainsMessage.h>
#include <Communication/Messages/Input/ChangeMasterMixInputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeMasterMixInputVolumesMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryMixInputVolumesMessage.h>
#include <Communication/Messages/Input/ChangeMasterOutputEqGainsMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryOutputEqGainsMessage.h>
#include <Communication/Messages/Input/ChangeMasterOutputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeAuxiliaryOutputVolumeMessage.h>
#include <Communication/Messages/Input/ChangeAllProcessingParametersMessage.h>
#include <Communication/Messages/Input/ListenProbeMessage.h>
#include <Communication/Messages/Input/StopProbeListeningMessage.h>
#include <Communication/Messages/Input/ToogleUniformizationMessage.h>

#include <Communication/Messages/Output/SoundErrorMessage.h>
#include <Communication/Messages/Output/InputSpectrumMessage.h>
#include <Communication/Messages/Output/SoundLevelMessage.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace adaptone;
using namespace nlohmann;
using namespace std;
using namespace ::testing;

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

DEFINE_TYPE_MATCHER(ChangeInputGainMessage);
DEFINE_TYPE_MATCHER(ChangeInputGainsMessage);
DEFINE_TYPE_MATCHER(ChangeInputEqGainsMessage);
DEFINE_TYPE_MATCHER(ChangeMasterMixInputVolumeMessage);
DEFINE_TYPE_MATCHER(ChangeMasterMixInputVolumesMessage);
DEFINE_TYPE_MATCHER(ChangeAuxiliaryMixInputVolumeMessage);
DEFINE_TYPE_MATCHER(ChangeAuxiliaryMixInputVolumesMessage);
DEFINE_TYPE_MATCHER(ChangeMasterOutputEqGainsMessage);
DEFINE_TYPE_MATCHER(ChangeAuxiliaryOutputEqGainsMessage);
DEFINE_TYPE_MATCHER(ChangeMasterOutputVolumeMessage);
DEFINE_TYPE_MATCHER(ChangeAuxiliaryOutputVolumeMessage);
DEFINE_TYPE_MATCHER(ChangeAllProcessingParametersMessage);
DEFINE_TYPE_MATCHER(ListenProbeMessage);
DEFINE_TYPE_MATCHER(StopProbeListeningMessage);
DEFINE_TYPE_MATCHER(ToogleUniformizationMessage);

DEFINE_TYPE_MATCHER(SoundErrorMessage);
DEFINE_TYPE_MATCHER(InputSpectrumMessage);
DEFINE_TYPE_MATCHER(SoundLevelMessage);

class ApplicationMessageHandlerMock : public ApplicationMessageHandler
{
public:
    ApplicationMessageHandlerMock()
    {}

    ~ApplicationMessageHandlerMock() override
    {}

    MOCK_METHOD2(handleDeserialized, void(const ApplicationMessage&, const function<void(const ApplicationMessage&)>&));
};

TEST(ApplicationMessageHandlerTests, handle_ConfigurationChoiceMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsConfigurationChoiceMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 0,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"inputChannelIds\": [1, 2, 3],"
        "    \"speakersNumber\": 4,"
        "    \"auxiliaryChannelIds\": [4, 5],"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\","
        "        \"id\": 5"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_InitialParametersCreationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsInitialParametersCreationMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 1,"
        "  \"data\": {"
        "    \"id\": 10,"
        "    \"name\": \"super nom\","
        "    \"inputChannelIds\": [1, 2, 3],"
        "    \"speakersNumber\": 4,"
        "    \"auxiliaryChannelIds\": [4, 5]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_LaunchInitializationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsLaunchInitializationMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 2"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_PositionConfirmationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsPositionConfirmationMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 3,"
        "  \"data\": {"
        "    \"firstSymmetryPositions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\","
        "        \"id\": 2"
        "      }"
        "    ],"
        "    \"secondSymmetryPositions\": ["
        "      {"
        "        \"x\": 340,"
        "        \"y\": 140,"
        "        \"type\": \"s\","
        "        \"id\": 3"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_RelaunchInitializationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsRelaunchInitializationMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 4"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_SymmetryConfirmationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsSymmetryConfirmationMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 5,"
        "  \"data\": {"
        "    \"symmetry\": 0"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_OptimizePositionMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsOptimizePositionMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 6"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_OptimizedPositionMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsOptimizedPositionMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 7,"
        "  \"data\": {"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\","
        "        \"id\": 5"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ReoptimizePositionMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsReoptimizePositionMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 8"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ConfigurationConfirmationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsConfigurationConfirmationMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 9"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeInputGainMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeInputGainMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 10,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gain\": 1.2"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_hangeInputGainsMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeInputGainsMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 11,"
        "  \"data\": {"
        "    \"gains\": ["
        "      {"
        "         \"channelId\": 1,"
        "         \"gain\" : 1"
        "      },"
        "      {"
        "        \"channelId\": 2,"
        "        \"gain\" : 10"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeInputEqGainsMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeInputEqGainsMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 12,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gains\": [1.0, 1.2, 1.23]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeMasterMixInputVolumeMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeMasterMixInputVolumeMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 13,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gain\": 1.0"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeMasterMixInputVolumesMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeMasterMixInputVolumesMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 14,"
        "  \"data\": {"
        "    \"gains\": ["
        "      {"
        "         \"channelId\": 1,"
        "         \"gain\" : 1"
        "      },"
        "      {"
        "        \"channelId\": 2,"
        "        \"gain\" : 10"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeAuxiliaryMixInputVolumeMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeAuxiliaryMixInputVolumeMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 15,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"auxiliaryChannelId\": 0,"
        "    \"gain\": 1.0"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeAuxiliaryMixInputVolumesMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeAuxiliaryMixInputVolumesMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 16,"
        "  \"data\": {"
        "    \"auxiliaryChannelId\": 1,"
        "    \"gains\": ["
        "      {"
        "         \"channelId\": 1,"
        "         \"gain\" : 1"
        "      },"
        "      {"
        "        \"channelId\": 2,"
        "        \"gain\" : 10"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeMasterOutputEqGainsMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeMasterOutputEqGainsMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 17,"
        "  \"data\": {"
        "    \"gains\": [1.0, 1.2, 1.23]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeAuxiliaryOutputEqGainsMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeAuxiliaryOutputEqGainsMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 18,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gains\": [1.0, 1.2, 1.23]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeMasterOutputVolumeMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeMasterOutputVolumeMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 19,"
        "  \"data\": {"
        "    \"gain\": 1.0"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeAuxiliaryOutputVolumeMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeAuxiliaryOutputVolumeMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 20,"
        "  \"data\": {"
        "    \"channelId\": 0,"
        "    \"gain\": 1.0"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ChangeAllProcessingParametersMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsChangeAllProcessingParametersMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 24,"
        "  \"data\": {"
        "    \"channels\":{"
        "      \"inputs\":["
        "        {"
        "          \"data\": {"
        "            \"channelId\":2,"
        "            \"gain\":3,"
        "            \"isMuted\":true,"
        "            \"isSolo\":false,"
        "            \"eqGains\": [2]"
        "          }"
        "        }"
        "      ],"
        "      \"master\":{"
        "        \"data\": {"
        "          \"gain\":3,"
        "          \"isMuted\":false,"
        "          \"inputs\":["
        "            {"
        "              \"data\": {"
        "                \"channelId\":1,"
        "                \"gain\":2"
        "              }"
        "            },"
        "            {"
        "              \"data\": {"
        "                \"channelId\":2,"
        "                \"gain\":3"
        "              }"
        "            }"
        "          ],"
        "          \"eqGains\": [3]"
        "        }"
        "      },"
        "      \"auxiliaries\":["
        "        {"
        "          \"data\": {"
        "            \"channelId\":10,"
        "            \"gain\":5,"
        "            \"isMuted\":false,"
        "            \"inputs\":["
        "              {"
        "                \"data\": {"
        "                  \"channelId\":1,"
        "                  \"gain\":0"
        "                }"
        "              },"
        "              {"
        "                \"data\": {"
        "                  \"channelId\":2,"
        "                  \"gain\":2"
        "                }"
        "              }"
        "            ],"
        "            \"eqGains\": [4]"
        "          }"
        "        }"
        "      ]"
        "    },"
        "    \"inputChannelIds\": [1, 2, 3],"
        "    \"speakersNumber\": 2,"
        "    \"auxiliaryChannelIds\": [4, 5]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ListenProbeMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsListenProbeMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 25,"
        "  \"data\": {"
        "    \"probeId\": 5"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_StopProbeListeningMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsStopProbeListeningMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 26"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_ToogleUniformizationMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsToogleUniformizationMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 27,"
        "  \"data\": {"
        "    \"isUniformizationOn\": true"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_SoundErrorMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsSoundErrorMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 21,"
        "  \"data\": {"
        "    \"positions\": ["
        "      {"
        "        \"x\": 140,"
        "        \"y\": 340,"
        "        \"type\": \"s\","
        "        \"errorRate\": 1"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_InputSpectrumMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsInputSpectrumMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 22,"
        "  \"data\": {"
        "    \"spectrums\": ["
        "      {"
        "        \"channelId\": 5,"
        "        \"points\": ["
        "          {"
        "            \"freq\": 1,"
        "            \"amplitude\": 2"
        "          }"
        "        ]"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}

TEST(ApplicationMessageHandlerTests, handle_SoundLevelMessage_shouldCallHandleWithTheRightType)
{
    ApplicationMessageHandlerMock applicationMessageHandler;
    EXPECT_CALL(applicationMessageHandler, handleDeserialized(IsSoundLevelMessage(), _));

    string serializedMessage = "{"
        "  \"seqId\": 23,"
        "  \"data\": {"
        "    \"inputAfterGain\": ["
        "      {"
        "        \"channelId\": 0,"
        "        \"level\": 1"
        "      },"
        "      {"
        "        \"channelId\": 1,"
        "        \"level\": 2"
        "      }"
        "    ],"
        "    \"inputAfterEq\": ["
        "      {"
        "        \"channelId\": 0,"
        "        \"level\": 3"
        "      },"
        "      {"
        "        \"channelId\": 1,"
        "        \"level\": 4"
        "      }"
        "    ],"
        "    \"outputAfterGain\": ["
        "      {"
        "        \"channelId\": 0,"
        "        \"level\": 5"
        "      },"
        "      {"
        "        \"channelId\": 1,"
        "        \"level\": 6"
        "      }"
        "    ]"
        "  }"
        "}";

    json j = json::parse(serializedMessage);
    applicationMessageHandler.handle(j, [](const ApplicationMessage&) {});
}
