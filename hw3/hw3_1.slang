import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float a = float (width)/ float (height); //aspect ratio

    // normalized coordinates u, v
    float u = (float(tid.x) + 0.5) / (float(width));
    float v = (float(tid.y) + 0.5) / (float(height));

    // image plane coordinates (xp, yp, -1)
    float xp = (2 * a * u) - a;
    float yp = 1 - (2 * v);

    float3 ray_origin = float3(0.0, 0.0, 0.0);
    float3 ray_dir = normalize(float3(xp, yp, -1.0));

    // track the closest valid intersection
    float t_min = 1e30;
    int hit_index = -1;

    for (int i = 0; i < num_spheres; i++) {
        Sphere sp = spheres[i];
        float3 oc = ray_origin - sp.center;

        float A = dot(ray_dir, ray_dir);
        float B = 2.0 * dot(ray_dir, oc);
        float C = dot(oc, oc) - sp.radius * sp.radius;

        float2 t_roots;
        if (solve_quadratic(A, B, C, t_roots)) {
            float t0 = t_roots.x;
            float t1 = t_roots.y;

            // find the closest positive intersection
            float t = -1.0;
            if (t0 > 0.0 && t1 > 0.0) {
                t = min(t0, t1);
            } else if (t0 > 0.0) {
                t = t0;
            } else if (t1 > 0.0) {
                t = t1;
            }

            if (t > 0.0 && t < t_min) {
                t_min = t;
                hit_index = i;
            }
        }
    }

    // shading
    if (hit_index >= 0) {
        Sphere sp = spheres[hit_index];
        float3 hit_pos = ray_origin + t_min * ray_dir;
        float3 normal = normalize((hit_pos - sp.center) / sp.radius);
        float3 color = (normal + 1.0) * 0.5;
        image[tid] = float4(color, 1.0);
    } else {
        image[tid] = float4(background, 1.0);
    }   
    
}
