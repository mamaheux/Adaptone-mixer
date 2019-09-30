#include <Communication/Messages/PositionType.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace nlohmann;

TEST(PositionTypeTests, serialization_shouldSerializaToJson)
{
    EXPECT_EQ(json(PositionType::Speaker), "s");
    EXPECT_EQ(json(PositionType::Probe), "m");
    EXPECT_THROW(json(static_cast<PositionType>(-1)), NotSupportedException);
}

TEST(PositionTypeTests, deserialization_shouldDeserializeFromJson)
{
    EXPECT_EQ(json("s").get<PositionType>(), PositionType::Speaker);
    EXPECT_EQ(json("m").get<PositionType>(), PositionType::Probe);
    EXPECT_THROW(json("asd").get<PositionType>(), NotSupportedException);
}

