__constant sampler_t TEX_SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;
__constant float THRESHOLD = 20.0f;

// Change these 2 defines to change precision
#define vec float3

#define s2(a, b)				temp = a; a = fmin(a, b); b = fmax(temp, b);
#define t2(a, b)				s2(v[a], v[b]);
#define t24(a, b, c, d, e, f, g, h)			t2(a, b); t2(c, d); t2(e, f); t2(g, h); 
#define t25(a, b, c, d, e, f, g, h, i, j)		t24(a, b, c, d, e, f, g, h); t2(i, j);


__kernel void median(
        read_only image2d_t src,
        write_only image2d_t dst
    ) 
{
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    vec v[25];
    
    const float4 central_color = read_imagef(src, TEX_SAMPLER, gid);

    if (central_color.w && central_color.w < THRESHOLD) {
        // Add the pixels which make up our window to the pixel array.
        for(int dX = -2; dX <= 2; ++dX) {
            for(int dY = -2; dY <= 2; ++dY) {		     
                const float4 color = read_imagef(src, TEX_SAMPLER, gid + (int2)(dX, dY));
                v[(dX + 2) * 5 + (dY + 2)] = color.xyz / color.w;
            }
        }

        vec temp;

        t25(0, 1,		3, 4,		2, 4,		2, 3,		6, 7);
        t25(5, 7,		5, 6,		9, 7,		1, 7,		1, 4);
        t25(12, 13,		11, 13,		11, 12,		15, 16,		14, 16);
        t25(14, 15,		18, 19,		17, 19,		17, 18,		21, 22);
        t25(20, 22,		20, 21,		23, 24,		2, 5,		3, 6);
        t25(0, 6,		0, 3,		4, 7,		1, 7,		1, 4);
        t25(11, 14,		8, 14,		8, 11,		12, 15,		9, 15);
        t25(9, 12,		13, 16,		10, 16,		10, 13,		20, 23);
        t25(17, 23,		17, 20,		21, 24,		18, 24,		18, 21);
        t25(19, 22,		8, 17,		9, 18,		0, 18,		0, 9);
        t25(10, 19,		1, 19,		1, 10,		11, 20,		2, 20);
        t25(2, 11,		12, 21,		3, 21,		3, 12,		13, 22);
        t25(4, 22,		4, 13,		14, 23,		5, 23,		5, 14);
        t25(15, 24,		6, 24,		6, 15,		7, 16,		7, 19);
        t25(3, 11,		5, 17,		11, 17,		9, 17,		4, 10);
        t25(6, 12,		7, 14,		4, 6,		4, 7,		12, 14);
        t25(10, 14,		6, 7,		10, 12,		6, 10,		6, 17);
        t25(12, 17,		7, 17,		7, 10,		12, 18,		7, 12);
        t24(10, 18,		12, 20,		10, 20,		10, 12);

        write_imagef(dst, gid, (float4)(v[12] * central_color.w, central_color.w));
    }
    else {
        write_imagef(dst, gid, central_color);
    }
}