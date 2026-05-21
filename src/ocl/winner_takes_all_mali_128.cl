@MAX_DISPARITY@
@NUM_PATHS@
#define INVALID_DISP (uint16_t)(-1)

inline uint packed_cost_index(uint cost, uint index)
{
    return (cost << 16) | (index & 0xffffu);
}

inline uint packed_cost(uint packed)
{
    return packed >> 16;
}

inline uint packed_index(uint packed)
{
    return packed & 0xffffu;
}

inline uint sum_costs_for_left_pixel(
    const global uint8_t* src,
    uint cost_step,
    uint base,
    uint disparity)
{
    uint sum = 0;
    for (uint p = 0; p < NUM_PATHS; ++p)
    {
        sum += src[p * cost_step + base + disparity];
    }
    return sum;
}

kernel void winner_takes_all_left_mali_128_kernel(
    global uint16_t* left_dest,
    const global uint8_t* src,
    int width,
    int height,
    int pitch,
    float uniqueness)
{
    const uint x = get_group_id(0);
    const uint y = get_group_id(1);
    const uint d = get_local_id(0);
    if (x >= (uint)width || y >= (uint)height)
    {
        return;
    }

    local uint packed_costs[MAX_DISPARITY];
    local uint unique_flags[MAX_DISPARITY];

    const uint cost_step = (uint)width * (uint)height * MAX_DISPARITY;
    const uint base = (y * (uint)width + x) * MAX_DISPARITY;
    const uint cost = sum_costs_for_left_pixel(src, cost_step, base, d);
    packed_costs[d] = packed_cost_index(cost, d);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = MAX_DISPARITY >> 1; stride > 0; stride >>= 1)
    {
        if (d < stride)
        {
            packed_costs[d] = min(packed_costs[d], packed_costs[d + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint best = packed_costs[0];
    const uint best_cost = packed_cost(best);
    const uint best_disp = packed_index(best);
    const bool near_best = abs((int)d - (int)best_disp) <= 1;
    unique_flags[d] = (near_best || (float)cost * uniqueness >= (float)best_cost) ? 1u : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = MAX_DISPARITY >> 1; stride > 0; stride >>= 1)
    {
        if (d < stride)
        {
            unique_flags[d] = min(unique_flags[d], unique_flags[d + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (d == 0)
    {
        left_dest[y * (uint)pitch + x] = unique_flags[0] ? (uint16_t)best_disp : INVALID_DISP;
    }
}

kernel void winner_takes_all_right_mali_128_kernel(
    global uint16_t* right_dest,
    const global uint8_t* src,
    int width,
    int height,
    int pitch)
{
    const uint x = get_group_id(0);
    const uint y = get_group_id(1);
    const uint d = get_local_id(0);
    if (x >= (uint)width || y >= (uint)height)
    {
        return;
    }

    local uint packed_costs[MAX_DISPARITY];
    const uint cost_step = (uint)width * (uint)height * MAX_DISPARITY;
    const uint left_x = x + d;
    if (left_x < (uint)width)
    {
        const uint base = (y * (uint)width + left_x) * MAX_DISPARITY;
        const uint cost = sum_costs_for_left_pixel(src, cost_step, base, d);
        packed_costs[d] = packed_cost_index(cost, d);
    }
    else
    {
        packed_costs[d] = 0xffffffffu;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = MAX_DISPARITY >> 1; stride > 0; stride >>= 1)
    {
        if (d < stride)
        {
            packed_costs[d] = min(packed_costs[d], packed_costs[d + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (d == 0)
    {
        const uint best = packed_costs[0];
        right_dest[y * (uint)pitch + x] =
            best == 0xffffffffu ? INVALID_DISP : (uint16_t)packed_index(best);
    }
}
