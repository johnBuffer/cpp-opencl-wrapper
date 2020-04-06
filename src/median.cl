__constant sampler_t TEX_SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;
__constant float THRESHOLD = 20000.0f;

// Change these 2 defines to change precision
#define vec float3

#define s2(a, b)        temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c)      s2(a, b); s2(a, c);
#define mx3(a, b, c)      s2(b, c); s2(a, c);

#define mnmx3(a, b, c)      mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)   s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)  s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges


__kernel void median(
        read_only image2d_t src,
        write_only image2d_t dst
    ) 
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    vec v[25];
    
    const float4 central_color = read_imagef(src, TEX_SAMPLER, gid);

    if (central_color.w && central_color.w < THRESHOLD) {
        vec v[6];
        float4 color = read_imagef(src, TEX_SAMPLER, gid + (int2)(-1, -1));
        v[0] = color.xyz / color.w;
        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(0, -1));
        v[1] = color.xyz / color.w;
        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(1, -1));
        v[2] = color.xyz / color.w;
        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(-1, 0));
        v[3] = color.xyz / color.w;
        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(0, 0));
        v[4] = color.xyz / color.w;
        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(1, 0));
        v[5] = color.xyz / color.w;

        // Starting with a subset of size 6, remove the min and max each time
        vec temp;
        mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);

        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(-1, 1));
        v[5] = color.xyz / color.w;

        mnmx5(v[1], v[2], v[3], v[4], v[5]);

        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(0, 1));
        v[5] = color.xyz / color.w;

        mnmx4(v[2], v[3], v[4], v[5]);

        color = read_imagef(src, TEX_SAMPLER, gid + (int2)(1, 1));
        v[5] = color.xyz / color.w;

        mnmx3(v[3], v[4], v[5]);

        write_imagef(dst, gid, (float4)(v[4], central_color.w));
    }
    else {
        write_imagef(dst, gid, central_color);
    }
}
