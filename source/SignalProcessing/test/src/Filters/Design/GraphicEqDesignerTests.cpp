#include <SignalProcessing/Filters/Design/GraphicEqDesigner.h>

#include <gtest/gtest.h>

#include <iostream>
#include <chrono>

using namespace adaptone;
using namespace std;

static constexpr size_t SampleFrequency = 48000;
static constexpr double MaxAbsError = 0.0001;

TEST(GraphicEqDesignerTests, constructor_notSortedCenterFrequencies_shouldThrowInvalidValueException)
{
    EXPECT_THROW(GraphicEqDesigner<float>(SampleFrequency, { 20, 10 }), InvalidValueException);
}

TEST(GraphicEqDesignerTests, constructor_emptyCenterFrequencies_shouldThrowInvalidValueException)
{
    EXPECT_THROW(GraphicEqDesigner<float>(SampleFrequency, {}), InvalidValueException);
}

TEST(GraphicEqDesignerTests, constructor_shouldInitializeThePoles)
{
    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };

    GraphicEqDesigner<float> designer(SampleFrequency, frequencies);

    ASSERT_EQ(designer.biquadCoefficients().size(), frequencies.size() * 2);

    EXPECT_NEAR(designer.biquadCoefficients()[0].a1, -1.9987, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a2, 0.9987, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[1].a1, -1.9992, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a2, 0.9992, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[2].a1, -1.9997, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[2].a2, 0.9997, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[3].a1, -1.9996, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[3].a2, 0.9996, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[4].a1, -1.9996, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[4].a2, 0.9996, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[5].a1, -1.9995, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[5].a2, 0.9995, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[6].a1, -1.9994, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[6].a2, 0.9994, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[7].a1, -1.9994, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[7].a2, 0.9994, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[8].a1, -1.9993, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[8].a2, 0.9993, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[9].a1, -1.9992, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[9].a2, 0.9992, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[10].a1, -1.9991, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[10].a2, 0.9991, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[11].a1, -1.9990, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[11].a2, 0.9990, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[12].a1, -1.9988, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[12].a2, 0.9989, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[13].a1, -1.9987, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[13].a2, 0.9988, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[14].a1, -1.9986, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[14].a2, 0.9987, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[15].a1, -1.9984, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[15].a2, 0.9985, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[16].a1, -1.9981, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[16].a2, 0.9984, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[17].a1, -1.9978, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[17].a2, 0.9980, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[18].a1, -1.9974, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[18].a2, 0.9977, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[19].a1, -1.9971, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[19].a2, 0.9975, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[20].a1, -1.9968, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[20].a2, 0.9974, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[21].a1, -1.9964, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[21].a2, 0.9971, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[22].a1, -1.9959, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[22].a2, 0.9967, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[23].a1, -1.9952, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[23].a2, 0.9962, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[24].a1, -1.9944, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[24].a2, 0.9958, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[25].a1, -1.9934, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[25].a2, 0.9951, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[26].a1, -1.9923, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[26].a2, 0.9945, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[27].a1, -1.9912, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[27].a2, 0.9940, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[28].a1, -1.9900, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[28].a2, 0.9935, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[29].a1, -1.9882, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[29].a2, 0.9925, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[30].a1, -1.9861, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[30].a2, 0.9915, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[31].a1, -1.9834, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[31].a2, 0.9902, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[32].a1, -1.9802, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[32].a2, 0.9889, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[33].a1, -1.9770, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[33].a2, 0.9880, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[34].a1, -1.9732, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[34].a2, 0.9870, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[35].a1, -1.9683, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[35].a2, 0.9854, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[36].a1, -1.9622, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[36].a2, 0.9838, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[37].a1, -1.9540, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[37].a2, 0.9806, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[38].a1, -1.9429, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[38].a2, 0.9774, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[39].a1, -1.9324, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[39].a2, 0.9758, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[40].a1, -1.9194, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[40].a2, 0.9742, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[41].a1, -1.9036, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[41].a2, 0.9710, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[42].a1, -1.8828, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[42].a2, 0.9678, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[43].a1, -1.8586, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[43].a2, 0.9631, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[44].a1, -1.8256, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[44].a2, 0.9583, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[45].a1, -1.7880, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[45].a2, 0.9521, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[46].a1, -1.7360, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[46].a2, 0.9459, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[47].a1, -1.6804, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[47].a2, 0.9413, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[48].a1, -1.6094, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[48].a2, 0.9366, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[49].a1, -1.5281, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[49].a2, 0.9275, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[50].a1, -1.4160, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[50].a2, 0.9184, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[51].a1, -1.2926, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[51].a2, 0.9065, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[52].a1, -1.1219, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[52].a2, 0.8947, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[53].a1, -0.9413, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[53].a2, 0.8860, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[54].a1, -0.7169, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[54].a2, 0.8773, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[55].a1, -0.4809, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[55].a2, 0.8631, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[56].a1, -0.1806, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[56].a2, 0.8491, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[57].a1, 0.1186, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[57].a2, 0.8217, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[58].a1, 0.5177, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[58].a2, 0.7953, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[59].a1, 0.8845, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[59].a2, 0.7824, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[60].a1, 1.2407, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[60].a2, 0.7697, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[61].a1, 1.5195, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[61].a2, 0.7697, MaxAbsError);
}

TEST(GraphicEqDesignerTests, update_invalidGainCount_shouldThrowInvalidValueException)
{
    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };

    vector<double> gainsDb{ 12, 2 };

    GraphicEqDesigner<double> designer(SampleFrequency, frequencies);
    EXPECT_THROW(designer.update(gainsDb), InvalidValueException);
}

TEST(GraphicEqDesignerTests, update_shouldSetTheRightCoefficents)
{
    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };

    vector<double> gainsDb{ 12, 11.5, 11, 9.5, 6, 4, 2, 1, 0, 6, 6, 12, 6, 6, -12, 12, -12, -12, -12, -12, 0, 0, 0, 0,
        -3, -6, -9, -12, 0, 0, 0 };

    GraphicEqDesigner<double> designer(SampleFrequency, frequencies);
    designer.update(gainsDb);

    ASSERT_EQ(designer.biquadCoefficients().size(), frequencies.size() * 2);

    for (size_t i = 0; i < designer.biquadCoefficients().size(); i++)
    {
        EXPECT_NEAR(designer.biquadCoefficients()[i].b2, 0, MaxAbsError);
    }

    EXPECT_NEAR(designer.biquadCoefficients()[0].b0, 0.0014, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b1, -0.0014, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[1].b0, 0.0003, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b1, -0.0003, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[2].b0, 0.0003, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[2].b1, -0.0003, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[3].b0, 0.0004, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[3].b1, -0.0003, MaxAbsError);

    //TODO continuer les EXPECT_NEAR

    EXPECT_NEAR(designer.d0(), 0.5251, MaxAbsError);
}

TEST(GraphicEqDesignerTests, performance)
{
    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };

    vector<double> gainsDb{ 12, 11.5, 11, 9.5, 6, 4, 2, 1, 0, 6, 6, 12, 6, 6, -12, 12, -12, -12, -12, -12, 0, 0, 0, 0,
        -3, -6, -9, -12, 0, 0, 0 };

    GraphicEqDesigner<double> designer(SampleFrequency, frequencies);

    constexpr std::size_t Count = 100;

    auto start = chrono::system_clock::now();
    for (std::size_t i = 0; i < Count; i++)
    {
        designer.update(gainsDb);
    }
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;

    cout << "Elapsed time = " << elapsed_seconds.count() / Count << " s" << endl;
}
