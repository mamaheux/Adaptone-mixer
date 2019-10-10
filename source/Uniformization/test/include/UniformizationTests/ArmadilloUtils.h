#ifndef UNIFORMIZATION_TESTS_ARMADILLO_UTILS_H
#define UNIFORMIZATION_TESTS_ARMADILLO_UTILS_H

#include <gtest/gtest.h>

#define EXPECT_MAT_NEAR(mat1, mat2, absError) \
    ASSERT_EQ((mat1).n_rows, (mat2).n_rows); \
    ASSERT_EQ((mat1).n_rows, (mat2).n_rows); \
    for (int i = 0; i < (mat1).n_rows; i++) \
    { \
        for (int j = 0; j < (mat1).n_cols; j++) \
        { \
            EXPECT_NEAR((mat1)(i,j), (mat2)(i,j), (absError)); \
        } \
    }

#endif
