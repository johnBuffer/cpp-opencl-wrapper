
__kernel void julia(
    float zoom,
    float2 pos,
    float2 param,
    __global unsigned char* out_image)
{
    //vec2 pos = gl_FragCoord.xy;
    const int2   gid              = (int2)(get_global_id(0), get_global_id(1));
    const int2   screen_size      = (int2)(get_global_size(0), get_global_size(1));
    const float2 screen_size_f    = (float2)(zoom * screen_size.x, zoom * screen_size.y);
    const float  screen_ratio     = screen_size_f.x / screen_size_f.y;
    const float2 c_space_coords   = convert_float2(gid) / max(screen_size_f.x, screen_size_f.y);

    const float max_it = 255.0f;

    float2 z = c_space_coords - (float2)(0.5f / zoom, 0.5f / (zoom * screen_ratio)) + pos;

    float i = 0.0f;
    
    while (z.x*z.x + z.y*z.y < 4.0f && i < max_it)
    {
        const float tmp = z.x;
        z.x = z.x*z.x - z.y*z.y + param.x;
        z.y = 2.0f*z.y*tmp + param.y;
        i += 1.0f;
    }

    float r=0.0f;
    float g=0.0f;
    float b=0.0f;

    const float t = max_it/3.0f;

    if (i < t) {
        r = i/t;
    }
    else if (i < 2.0f*t) {
        r = 1.0f;
        g = (i-t)/t;
    }
    else if (i <= max_it) {
        r = 1.0f;
        g = 1.0f;
        b = (i-2.0f*t)/t;
    }
    
    const unsigned int index = 4 * (gid.x + gid.y * screen_size.x);
	out_image[index + 0] = (unsigned int)(255.0f*r);
	out_image[index + 1] = (unsigned int)(255.0f*g);
	out_image[index + 2] = (unsigned int)(255.0f*b);
	out_image[index + 3] = 255;
}