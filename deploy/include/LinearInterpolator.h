// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <vector>
#include <cassert>

inline std::vector<float> linear_interpolate(float t, const std::vector<float>& ts, const std::vector<std::vector<float>>& ys)
{
    assert(ts.size() == ys.size() && !ys.empty() && ts.size() > 1 && ys[0].size() > 0);

    if (t <= ts[0]) return ys[0];
    if (t >= ts[ts.size() - 1]) return ys[ts.size() - 1];

    for (int i = 0; i < ts.size() - 1; ++i)
    {
        if (t >= ts[i] && t <= ts[i + 1])
        {
            float alpha = (t - ts[i]) / (ts[i + 1] - ts[i]);
            std::vector<float> result(ys[i].size());
            for (int j = 0; j < ys[i].size(); ++j)
            {
                result[j] = ys[i][j] * (1 - alpha) + ys[i + 1][j] * alpha;
            }
            return result;
        }
    }

    return std::vector<float>(ys[0].size(), 0.0f); // Fallback, should not reach here
}
