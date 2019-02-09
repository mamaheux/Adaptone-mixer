#ifndef MIXER_TESTS_JETSON_TEST_H
#define MIXER_TESTS_JETSON_TEST_H

#include <gtest/gtest.h>

#ifdef JETSON

#define JETSON_TEST(testCase, test) TEST(testCase, test)

#else

#define JETSON_TEST(testCase, test) TEST(testCase, DISABLED_##test)

#endif

#endif
