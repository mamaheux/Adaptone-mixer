#include <Mixer/ChannelIdMapping.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(ChannelIdMappingTests, update_invalidParameters_shouldThrowInvalidValueException)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    EXPECT_THROW(mapping.update({ 0, 1, 2, 3, 4 }, {}, 1), InvalidValueException);
    EXPECT_THROW(mapping.update({ 0, 1 }, { 2, 3 }, 1), InvalidValueException);
    EXPECT_THROW(mapping.update({ 0, 1, 2, 3, 4 }, { 1 }, 2), InvalidValueException);
}

TEST(ChannelIdMappingTests, getChannelIdFromInputIndex_invalidIndex_shouldThrowInvalidValueException)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, {}, 1);

    EXPECT_ANY_THROW(mapping.getChannelIdFromInputIndex(4));
}

TEST(ChannelIdMappingTests, getChannelIdFromAuxiliaryOutputIndex_invalidIndex_shouldThrowInvalidValueException)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, { 4 }, 1);

    EXPECT_ANY_THROW(mapping.getChannelIdFromAuxiliaryOutputIndex(2));
}

TEST(ChannelIdMappingTests, getInputIndexFromChannelId_invalidChannelId_shouldThrowInvalidValueException)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, { 4 }, 1);

    EXPECT_ANY_THROW(mapping.getInputIndexFromChannelId(4));
}

TEST(ChannelIdMappingTests, getAuxiliaryOutputIndexFromChannelId_invalidChannelId_shouldThrowInvalidValueException)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, { 4 }, 1);

    EXPECT_ANY_THROW(mapping.getAuxiliaryOutputIndexFromChannelId(3));
}

TEST(ChannelIdMappingTests, getChannelIdFromInputIndexOrNull_invalidIndex_shouldReturnNull)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, {}, 1);

    EXPECT_EQ(mapping.getChannelIdFromInputIndexOrNull(4), nullopt);
}

TEST(ChannelIdMappingTests, getChannelIdFromAuxiliaryOutputIndexOrNull_invalidIndex_shouldReturnNull)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, { 4 }, 1);

    EXPECT_EQ(mapping.getChannelIdFromAuxiliaryOutputIndexOrNull(2), nullopt);
}

TEST(ChannelIdMappingTests, getInputIndexFromChannelIdOrNull_invalidChannelId_shouldReturnNull)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, { 4 }, 1);

    EXPECT_EQ(mapping.getInputIndexFromChannelIdOrNull(4), nullopt);
}

TEST(ChannelIdMappingTests, getAuxiliaryOutputIndexFromChannelIdOrNull_invalidChannelId_shouldReturnNull)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, { 4 }, 1);

    EXPECT_EQ(mapping.getAuxiliaryOutputIndexFromChannelIdOrNull(3), nullopt);
}

TEST(ChannelIdMappingTests, getChannelIdFromInputIndex_shouldReturnChannelId)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 1, 2, 3, 4 }, {}, 1);

    EXPECT_EQ(mapping.getChannelIdFromInputIndex(0), 1);
    EXPECT_EQ(mapping.getChannelIdFromInputIndex(1), 2);
    EXPECT_EQ(mapping.getChannelIdFromInputIndex(2), 3);
    EXPECT_EQ(mapping.getChannelIdFromInputIndex(3), 4);
}

TEST(ChannelIdMappingTests, getChannelIdFromAuxiliaryOutputIndex_shouldReturnChannelId)
{
    constexpr size_t InputChannelCount = 3;
    constexpr size_t OutputChannelCount = 4;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2 }, { 3, 4 }, 1);

    EXPECT_EQ(mapping.getChannelIdFromAuxiliaryOutputIndex(3), 3);
    EXPECT_EQ(mapping.getChannelIdFromAuxiliaryOutputIndex(1), 4);
}

TEST(ChannelIdMappingTests, getInputIndexFromChannelId_shouldReturnInputIndex)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 1, 2, 3, 4 }, { 4 }, 1);

    EXPECT_EQ(mapping.getInputIndexFromChannelId(1), 0);
    EXPECT_EQ(mapping.getInputIndexFromChannelId(2), 1);
    EXPECT_EQ(mapping.getInputIndexFromChannelId(3), 2);
    EXPECT_EQ(mapping.getInputIndexFromChannelId(4), 3);
}

TEST(ChannelIdMappingTests, getAuxiliaryOutputIndexFromChannelId_shouldReturnAuxiliaryOutputIndex)
{
    constexpr size_t InputChannelCount = 3;
    constexpr size_t OutputChannelCount = 4;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2 }, { 3, 4 }, 1);

    EXPECT_EQ(mapping.getAuxiliaryOutputIndexFromChannelId(3), 3);
    EXPECT_EQ(mapping.getAuxiliaryOutputIndexFromChannelId(4), 1);
}

TEST(ChannelIdMappingTests, getChannelIdFromInputIndexOrNull_shouldReturnChannelId)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 1, 2, 3, 4 }, {}, 1);

    EXPECT_EQ(mapping.getChannelIdFromInputIndexOrNull(0), 1);
    EXPECT_EQ(mapping.getChannelIdFromInputIndexOrNull(1), 2);
    EXPECT_EQ(mapping.getChannelIdFromInputIndexOrNull(2), 3);
    EXPECT_EQ(mapping.getChannelIdFromInputIndexOrNull(3), 4);
}

TEST(ChannelIdMappingTests, getChannelIdFromAuxiliaryOutputIndexOrNull_shouldReturnChannelId)
{
    constexpr size_t InputChannelCount = 3;
    constexpr size_t OutputChannelCount = 4;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2 }, { 3, 4 }, 1);

    EXPECT_EQ(mapping.getChannelIdFromAuxiliaryOutputIndexOrNull(3), 3);
    EXPECT_EQ(mapping.getChannelIdFromAuxiliaryOutputIndexOrNull(1), 4);
}

TEST(ChannelIdMappingTests, getInputIndexFromChannelIdOrNull_shouldReturnInputIndex)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 3;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 1, 2, 3, 4 }, { 4 }, 1);

    EXPECT_EQ(mapping.getInputIndexFromChannelIdOrNull(1), 0);
    EXPECT_EQ(mapping.getInputIndexFromChannelIdOrNull(2), 1);
    EXPECT_EQ(mapping.getInputIndexFromChannelIdOrNull(3), 2);
    EXPECT_EQ(mapping.getInputIndexFromChannelIdOrNull(4), 3);
}

TEST(ChannelIdMappingTests, getAuxiliaryOutputIndexFromChannelIdOrNull_shouldReturnAuxiliaryOutputIndex)
{
    constexpr size_t InputChannelCount = 3;
    constexpr size_t OutputChannelCount = 4;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2 }, { 3, 4 }, 1);

    EXPECT_EQ(mapping.getAuxiliaryOutputIndexFromChannelIdOrNull(3), 3);
    EXPECT_EQ(mapping.getAuxiliaryOutputIndexFromChannelIdOrNull(4), 1);
}

TEST(ChannelIdMappingTests, getMasterOutputIndexes_shouldMasterOutputIndexes)
{
    constexpr size_t InputChannelCount = 4;
    constexpr size_t OutputChannelCount = 6;
    const vector<size_t> HeadphoneChannelIndexes{ 2 };

    ChannelIdMapping mapping(InputChannelCount, OutputChannelCount, HeadphoneChannelIndexes);

    mapping.update({ 0, 1, 2, 3 }, { 4, 5 }, 3);

    EXPECT_EQ(mapping.getMasterOutputIndexes(), vector<size_t>({0, 1, 3}));
}
