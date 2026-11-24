// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <cassert>
#include <vector>

inline std::vector<float> linear_interpolate(
    float t,
    const std::vector<float>& ts,
    const std::vector<std::vector<float>>& ys)
{
    assert(ts.size() == ys.size() && ts.size() > 1 && !ys[0].empty());

    if (t <= ts.front()) return ys.front();
    if (t >= ts.back()) return ys.back();

    for (size_t i = 0; i + 1 < ts.size(); ++i)
    {
        if (t >= ts[i] && t <= ts[i + 1])
        {
            const float denom = ts[i + 1] - ts[i];
            if (denom == 0.0f) return ys[i + 1];

            const float alpha = (t - ts[i]) / denom;
            std::vector<float> result(ys[i].size());
            for (size_t j = 0; j < ys[i].size(); ++j)
            {
                result[j] = ys[i][j] * (1.0f - alpha) + ys[i + 1][j] * alpha;
            }
            return result;
        }
    }

    return ys.back();
}
