#include <gtest/gtest.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
extern "C" {
#include "../gglib/gg_math.h"
}


TEST(GgMathTest, computeAveStd_sinx) {
    size_t n_samples = 100'000;
    double omega = 2*M_PI / static_cast<double>(n_samples);

    std::vector<double> sinx(n_samples);
    for(size_t i = 0; i < n_samples; i++) {
        double theta = static_cast<double>(i);
        sinx[i] = sin(omega * theta);
    }

    double mean, var;
    computeAveStd(sinx.data(), sinx.size(), &mean, &var);

    EXPECT_NEAR(mean, 0.0, 1e-12);
    EXPECT_NEAR(var, 0.5, 1e-12);
}


TEST(GgMathTest, randn_checkMeanAndVar) {
    size_t n_samples = 100'000;
    std::vector<double> samples(n_samples);

    for(int i=0; i< samples.size(); i++){
        samples[i] = randn();
    }
    double mean, var;
    computeAveStd(samples.data(), n_samples, &mean, &var);
    EXPECT_NEAR(mean, 0.0, 0.05);     // Should be near 0
    EXPECT_NEAR(var, 1.0, 0.05);      // Should be near 1
}

TEST(GgMathTest, computePairwiseDistancesWithPCB_2ParticleAlongXnoPCB) {
    // Two particles along x-axis, no wrapping
    double coords[] = {0,0,0, 1,0,0};
    double r_utb[1];
    computePairwiseDistancesWithPCB(coords, 2, 100.0, r_utb);
    EXPECT_DOUBLE_EQ(r_utb[0], 1.0);
}

TEST(GgMathTest, computePairwiseDistancesWithPCB_2ParticleAlongXwithPCB) {
    // Two particles separated by almost box length → should wrap
    double L = 10.0;
    double coords[] = {0,0,0, 9.5,0,0};
    double r_utb[1];
    computePairwiseDistancesWithPCB(coords, 2, L, r_utb);
    EXPECT_NEAR(r_utb[0], 0.5, 1e-12);
}

TEST(GgMathTest, computePairwiseDistancesWithPCB_3ParticlesOnTriangle) {
    // 3 particles in a line at (0,0,0), (1,0,0), (2,0,0)
    double coords[] = {0,0,0, 1,0,0, 1,1,0};
    double r_utb[3]; // upper-triangular: (0,1),(0,2),(1,2)
    computePairwiseDistancesWithPCB(coords, 3, 100.0, r_utb);
    EXPECT_DOUBLE_EQ(r_utb[0], 1.0);
    EXPECT_DOUBLE_EQ(r_utb[1], sqrt(2));
    EXPECT_DOUBLE_EQ(r_utb[2], 1.0);
}
