import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
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

    float3 color = float3(0.0);
    float3 throughput = float3(1.0);

    // iterative reflection bounce loop
    for (int b = 0; b < max_depth; b++) {

        // track the closest valid intersection
        float t_min = 1e30;
        int hit_index = -1;

        for (int i = 0; i < num_shapes; i++) {
            Sphere sp = spheres[shapes[i].index];
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
                if (t0 > EPSILON && t1 > EPSILON) {
                    t = min(t0, t1);
                } else if (t0 > EPSILON) {
                    t = t0;
                } else if (t1 > EPSILON) {
                    t = t1;
                }

                if (t > EPSILON && t < t_min) {
                    t_min = t;
                    hit_index = i;
                }
            }
        }

        // ray missed the scene: accumulate background color through remaining throughput
        if (hit_index < 0) {
            color += throughput * background;
            break;
        }

        // compute surface hit attributes
        Shape hit_shape = shapes[hit_index];
        Sphere hit_sphere = spheres[hit_shape.index];

        float3 hit_pos = ray_origin + t_min * ray_dir;
        float3 normal = normalize((hit_pos - hit_sphere.center) / hit_sphere.radius);

        // shading with point lights and hard shadows
        float3 direct_light = float3(0.0, 0.0, 0.0);

        for (int l = 0; l < num_lights; l++) {
            PointLight light = lights[l];
            float3 to_light = light.position - hit_pos;
            float dist_light = length(to_light);
            float3 l_dir = to_light / dist_light;

            // shadow ray intersection test
            bool in_shadow = false;
            for (int s = 0; s < num_shapes; s++) {
                Sphere sp = spheres[shapes[s].index];
                float3 oc = hit_pos - sp.center;

                float A = dot(l_dir, l_dir);
                float B = 2.0 * dot(l_dir, oc);
                float C = dot(oc, oc) - sp.radius * sp.radius;

                float2 t_roots;
                if (solve_quadratic(A, B, C, t_roots)) {
                    float t0 = t_roots.x;
                    float t1 = t_roots.y;

                    if ((t0 > EPSILON && t0 < dist_light) || (t1 > EPSILON && t1 < dist_light)) {
                        in_shadow = true;
                        break;
                    }
                }
            }

            // if not occluded, add diffuse reflection
            if (!in_shadow) {
                float n_dot_l = max(0.0, dot(normal, l_dir));
                float3 intensity = light.intensity / (dist_light * dist_light);
                direct_light += hit_shape.Kd * intensity * n_dot_l;
            }
        }

        // accumulate direct lighting at this hit point
        color += throughput * direct_light;

        // if no mirror reflection, terminate bounce loop
        if (max(hit_shape.Kr.r, max(hit_shape.Kr.g, hit_shape.Kr.b)) <= 0.0) {
            break;
        }

        // update throughput and prepare next reflected ray
        throughput *= hit_shape.Kr;
        ray_origin = hit_pos;
        ray_dir = reflect(ray_dir, normal);
    }

    image[tid] = float4(color, 1.0);

}
