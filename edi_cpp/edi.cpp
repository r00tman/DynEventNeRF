// Fast event accumulation with decay (Rudnev et al., CVPRW 2025)
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>


namespace py = pybind11;

struct event_t {
    float t;
    int16_t x, y;
    int8_t p; // 1, -1, and 0 for special cases
};

struct accum_t {
    float t;
    // int n_pos, n_neg;
    float value;
};

class EventStoragePixel {
private:
    std::vector<accum_t> accums;

public:
    float get_val(float t) {
        if (accums.size() == 0) {
            // return std::make_pair(0, 0);
            return 0.0f;
        }
        // std::cout << "array: ";
        // for (auto &a: accums) {
        //     std::cout << a.t << " (" << a.n_pos << "," << a.n_neg << "), ";
        // }
        // std::cout << std::endl;

        int l = 0, r = accums.size();
        while (r - l > 1) {
            int m = (l + r) / 2;
            // float rval = (r>=accums.size())?1234:accums[r].t;
            // std::cout << "l=" << l << " (" << accums[l].t << ") r=" << r << " (" << rval << ")" << " m=" << m << " (" << accums[m].t << ")" << std::endl;
            if (accums[m].t > t) {
                r = m;
            } else {
                l = m;
            }
        }
        auto &accum = accums[l];
        // std::cout << "query " << t << ": found " << l << " (" << accum.t << ", " << accum.n_pos << ", " << accum.n_neg << ")" << std::endl;
        // return std::make_pair(accum.n_pos, accum.n_neg);
        return accum.value;
    }

    float get_diff(float startt, float endt) {
        auto startv = get_val(startt);
        auto endv = get_val(endt);
        // return std::make_pair(endv.first-startv.first, endv.second-startv.second);
        return endv-startv;
    }

    void add_event(float t, int8_t p, float decay_strength) {
        if (accums.size() == 0) {
            accums.push_back(accum_t{-1.0f, 0});
        }
        auto &last_accum = accums[accums.size()-1];
        // accum_t new_accum{t, last_accum.n_pos, last_accum.n_neg};
        accum_t new_accum{t, 0};
        if (p > 0) {
            // new_accum.n_pos += 1;
            new_accum.value = last_accum.value*decay_strength + 1;
        } else if (p < 0) {
            // new_accum.n_neg += 1;
            new_accum.value = last_accum.value*decay_strength - 1;
        }
        accums.emplace_back(new_accum);
    }
};

class EventStorage {
private:
    std::vector<std::vector<EventStoragePixel> > _pixels;
    int _H, _W;
    float _decay_strength;

public:
    EventStorage(int H, int W, float decay_strength, torch::Tensor xs, torch::Tensor ys, torch::Tensor ts, torch::Tensor ps) {
        _H = H;
        _W = W;
        _decay_strength = decay_strength;
        _pixels.clear();
        for (int i = 0; i < _H; ++i) {
            _pixels.push_back(std::vector<EventStoragePixel>(_W));
        }
        auto xs_a = xs.accessor<int16_t,1>();
        auto ys_a = ys.accessor<int16_t,1>();
        auto ts_a = ts.accessor<float,1>();
        auto ps_a = ps.accessor<int8_t,1>();
        auto N = ts.size(0);

        for (int i = 0; i < N; ++i) {
            float t = ts_a[i];
            int16_t x = xs_a[i];
            int16_t y = ys_a[i];
            int8_t p = ps_a[i];
            _pixels[y][x].add_event(t, p, _decay_strength);
        }
    }

    torch::Tensor accumulate(float start_t, float end_t) {
        torch::Tensor res;
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        res = torch::zeros({_H, _W}, options);

        auto res_a = res.accessor<float,2>();

        for (int i = 0; i < _H; ++i) {
            for (int j = 0; j < _W; ++j) {
                // pos, neg
                // std::pair<int, int> diff = _pixels[i][j].get_diff(start_t, end_t);
                // res_a[i][j][0] = diff.first;
                // res_a[i][j][1] = diff.second;
                float diff = _pixels[i][j].get_diff(start_t, end_t);
                res_a[i][j] = diff;
            }
        }

        return res;
    }
};

// Define the PyTorch module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<EventStorage>(m, "EventStorage")
        .def(py::init<int, int, float, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>())
        .def("accumulate", &EventStorage::accumulate, "Accumulate events");
}
