@DIRECTION@

#define feature_type uint32_t
#define MAX_DISPARITY 128u
kernel void aggregate_vertical_path_mali_128_kernel(
    global uint8_t* dest,
    global const feature_type* left,
    global const feature_type* right,
    int width,
    int height,
    unsigned int p1,
    unsigned int p2,
    int min_disp)
{
    const uint x = get_global_id(0);
    if (x >= (uint)width || width == 0 || height == 0)
    {
        return;
    }

    uint dp[MAX_DISPARITY];
    for (uint i = 0; i < MAX_DISPARITY; ++i)
    {
        dp[i] = 0;
    }
    uint last_min = 0;

    for (uint iter = 0; iter < (uint)height; ++iter)
    {
        const uint y = (DIRECTION > 0) ? iter : (uint)(height - 1) - iter;
        const feature_type left_value = left[x + y * (uint)width];

        uint new_dp[MAX_DISPARITY];
        uint row_min = 0xffffffffu;

        for (uint d = 0; d < MAX_DISPARITY; ++d)
        {
            const int right_x = (int)x - min_disp - (int)d;
            feature_type right_value = 0;
            if (0 <= right_x && right_x < width)
            {
                right_value = right[(uint)right_x + y * (uint)width];
            }
            const uint cost = popcount(left_value ^ right_value);

            uint best = min(dp[d] - last_min, p2);
            if (d > 0)
            {
                best = min(best, dp[d - 1] - last_min + p1);
            }

            if (d + 1 < MAX_DISPARITY)
            {
                best = min(best, dp[d + 1] - last_min + p1);
            }

            new_dp[d] = best + cost;
            row_min = min(row_min, new_dp[d]);
        }
        last_min = row_min;

        for (uint d = 0; d < MAX_DISPARITY; ++d)
        {
            dp[d] = new_dp[d];
            dest[d + x * MAX_DISPARITY + y * MAX_DISPARITY * (uint)width] =
                (uint8_t)min(dp[d], 255u);
        }
    }
}
