// accumulator_test: just isolated accumulator code & self-tests
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>

struct event_t {
    float t;
    int16_t x, y;
    int8_t p;  // 1, -1, and 0 for special cases
};

float C_pos = 1, C_neg = -1;

struct accum_t {
    float t;
    int n_pos, n_neg;

    float get_I() {
        return exp(C_pos*n_pos + C_neg*n_neg);
    }

    float diff_C_pos() {
        return n_pos * get_I();
    }

    float diff_C_neg() {
        return n_neg * get_I();
    }
};

class Accumulator {
public:
    std::vector<accum_t> accums;
    float full_value = 0;

    float C_pos_grad = 0;
    float C_neg_grad = 0;

    void process_event(const event_t &e) {
        if (accums.empty()) {
            assert(e.p == 0);
            accums.emplace_back(accum_t{e.t, 0, 0});
            return;
        }

        accum_t last_accum = accums[accums.size()-1];

        accum_t new_accum = last_accum;
        new_accum.t = e.t;
        if (e.p > 0) {
            new_accum.n_pos += 1;
        } else if (e.p < 0) {
            new_accum.n_neg += 1;
        }

        full_value += (e.t - last_accum.t) * last_accum.get_I();
        C_pos_grad += (e.t - last_accum.t) * last_accum.diff_C_pos();
        C_neg_grad += (e.t - last_accum.t) * last_accum.diff_C_neg();

        accums.emplace_back(new_accum);
    }

    void recompute() {
        full_value = 0;
        C_pos_grad = 0;
        C_neg_grad = 0;

        for (int i = 1; i < accums.size(); ++i) {
            accum_t last_accum = accums[i-1];
            accum_t new_accum = accums[i];

            full_value += (new_accum.t - last_accum.t) * last_accum.get_I();
            C_pos_grad += (new_accum.t - last_accum.t) * last_accum.diff_C_pos();
            C_neg_grad += (new_accum.t - last_accum.t) * last_accum.diff_C_neg();
        }
    }

    float get_I0(float B, float T0, float T1) {
        // B is the actual blurry value
        // full_value is the simulated blurry value, non-normalized
        // T0, T1 are the start and end times of the window generating B
        assert(T0 == accums[0].t);
        assert(T1 == accums[accums.size()-1].t);
        float result = B*(T1-T0)/full_value;
        return result;
    }

    float get_It(float t, float B, float T0, float T1) {
        float I0 = get_I0(B, T0, T1);

        float It_norm = 0;

        for (int i = 0; i < accums.size(); ++i) {
            if (i+1 >= accums.size() || accums[i+1].t > t) {
                assert(accums[i].t <= t);
                It_norm = accums[i].get_I();
                break;
            }
        }

        return It_norm * I0;

    }

};

int main(int argc, char *argv[]) {
    /* std::vector<event_t> events{ */
    /*     {1, 0, 0, +1}, {3, 0, 0, +1}, {4, 0, 0, -1} */
    /* }; */
    /* std::vector<event_t> events{ */
    /*     {1, 0, 0, +1}, {1, 0, 0, -1} */
    /* }; */
    std::vector<event_t> events{
        {1, 0, 0, +1}, {2, 0, 0, +1}, {3, 0, 0, -1}, {4, 0, 0, -1}
    };
    float T0 = 0, T1 = 5;

    const int W = 346;
    const int H = 260;

    std::vector<std::vector<Accumulator> > acc;

    for (int i = 0; i < H; ++i) {
        acc.push_back(std::vector<Accumulator>());
        for (int j = 0; j < W; ++j) {
            acc[i].push_back(Accumulator());
        }
    }

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            acc[i][j].process_event(event_t{T0, 0, 0, 0});
        }
    }

    for (int i = 0; i < events.size(); ++i) {
        acc[events[i].y][events[i].x].process_event(events[i]);
    }

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            acc[i][j].process_event(event_t{T1, 0, 0, 0});
        }
    }

    std::cout << "B = " << acc[0][0].full_value / (T1 - T0) << std::endl;
    std::cout << "dB/dC+ = " << acc[0][0].C_pos_grad / (T1 - T0) << std::endl;
    std::cout << "dB/dC- = " << acc[0][0].C_neg_grad / (T1 - T0) << std::endl;

    std::cout << std::endl;

    acc[0][0].recompute();

    std::cout << "B = " << acc[0][0].full_value / (T1 - T0) << std::endl;
    std::cout << "dB/dC+ = " << acc[0][0].C_pos_grad / (T1 - T0) << std::endl;
    std::cout << "dB/dC- = " << acc[0][0].C_neg_grad / (T1 - T0) << std::endl;

    std::cout << std::endl;
    /* float B = 3.30878; */
    float B = 2.96;
    /* float B = acc[0][0].full_value / (T1 - T0); */
    std::cout << "B' = " << B << std::endl;
    std::cout << "I0 = " << acc[0][0].get_I0(B, T0, T1) << std::endl;
    for (float t: {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}) {
        std::cout << "I(" << std::setw(3) << std::left << t << ") = " << acc[0][0].get_It(t, B, T0, T1) << std::endl;
    }

    C_pos = 1.1;
    C_neg = -0.1;

    float LR = 1e-3;
    for (int it = 0; it < 100; ++it) {
        acc[0][0].recompute();

        float B_est = acc[0][0].full_value / (T1 - T0);
        float loss = std::pow(B_est-B, 2);

        float dB_estdC_pos = acc[0][0].C_pos_grad / (T1 - T0);
        float dloss_dC_pos = 2*(B_est-B)*dB_estdC_pos;

        float dB_estdC_neg = acc[0][0].C_neg_grad / (T1 - T0);
        float dloss_dC_neg = 2*(B_est-B)*dB_estdC_neg;

        std::cout << "-------------------------------------" << std::endl;
        std::cout << "-- iteration " << it << " --" << std::endl;
        std::cout << "C_pos = " << C_pos << std::endl;
        std::cout << "C_neg = " << C_neg << std::endl;
        std::cout << "B_est = " << B_est << std::endl;
        std::cout << "B = " << B << std::endl;
        std::cout << "loss = " << loss << std::endl;
        std::cout << "dloss/dC+ = " << dloss_dC_pos << std::endl;
        std::cout << "dloss/dC- = " << dloss_dC_neg << std::endl;
        std::cout << "dB/dC+ = " << dB_estdC_pos << std::endl;
        std::cout << "dB/dC- = " << dB_estdC_pos << std::endl;
        C_pos -= LR * dloss_dC_pos;
        C_neg -= LR * dloss_dC_neg;
    }


    std::cout << std::endl;
    B = 3.30878;
    /* float B = acc[0][0].full_value / (T1 - T0); */
    std::cout << "B' = " << B << std::endl;
    std::cout << "I0 = " << acc[0][0].get_I0(B, T0, T1) << std::endl;
    for (float t: {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}) {
        std::cout << "I(" << std::setw(3) << std::left << t << ") = " << acc[0][0].get_It(t, B, T0, T1) << std::endl;
    }


/*     std::cout << "I(0) = " << acc[0][0].get_It(0, B, T0, T1) << std::endl; */
/*     std::cout << "I(1) = " << acc[0][0].get_It(1, B, T0, T1) << std::endl; */
/*     std::cout << "I(2) = " << acc[0][0].get_It(2, B, T0, T1) << std::endl; */
/*     std::cout << "I(3) = " << acc[0][0].get_It(3, B, T0, T1) << std::endl; */
/*     std::cout << "I(4) = " << acc[0][0].get_It(4, B, T0, T1) << std::endl; */
/*     std::cout << "I(5) = " << acc[0][0].get_It(5, B, T0, T1) << std::endl; */
/*     std::cout << "I(6) = " << acc[0][0].get_It(6, B, T0, T1) << std::endl; */

    return 0;
}
