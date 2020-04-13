__constant sampler_t TEX_SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;
__constant float THRESHOLD = 50.0f;


__kernel void test()
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    
    
}
