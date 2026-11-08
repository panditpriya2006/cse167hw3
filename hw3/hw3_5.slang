import hw3_common;

// Intersect a ray with every triangle in one mesh.
// The ray is already transformed into the mesh's object space.
//
// Returns:
//   true         if a triangle was hit
//   closest_t    the closest intersection distance
//   normal_obj   the interpolated vertex normal at the hit
bool intersect_mesh(
    TriangleMesh mesh,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<float3> vertex_normals,
    StructuredBuffer<uint3> triangles,
    float3 ray_org,
    float3 ray_dir,
    float t_max,
    out float closest_t,
    out float3 normal_obj
) {
    closest_t = t_max;
    normal_obj = float3(0.0);

    bool hit = false;

    // Check every triangle belonging to this mesh.
    for (uint j = mesh.triangle_begin; j < mesh.triangle_end; j++) {

        // Triangle indices are local to the mesh.
        // Add vertex_offset to get indices in the global arrays.
        uint3 tri = triangles[j] + mesh.vertex_offset;

        float3 p0 = vertices[tri.x];
        float3 p1 = vertices[tri.y];
        float3 p2 = vertices[tri.z];

        // Barycentric coordinates and ray intersection distance.
        float3 bary;
        float t;

        // Only accept intersections closer than the current closest hit.
        if (intersect_tri(
            p0, p1, p2,
            ray_org, ray_dir,
            EPSILON, closest_t,
            bary, t
        )) {
            closest_t = t;
            hit = true;

            // Interpolate the three vertex normals using barycentric weights.
            float3 n0 = vertex_normals[tri.x];
            float3 n1 = vertex_normals[tri.y];
            float3 n2 = vertex_normals[tri.z];

            normal_obj = normalize(
                bary.x * n0 +
                bary.y * n1 +
                bary.z * n2
            );
        }
    }

    return hit;
}


[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(
    uint2 tid : SV_DispatchThreadID,
    uniform float3 background,
    uniform Camera camera,
    StructuredBuffer<Shape> shapes,
    uniform int num_shapes,
    StructuredBuffer<Sphere> spheres,
    StructuredBuffer<TriangleMesh> meshes,
    StructuredBuffer<float3> vertices,
    StructuredBuffer<float3> vertex_normals,
    StructuredBuffer<uint3> triangles,
    StructuredBuffer<PointLight> lights,
    uniform int num_lights,
    uniform int max_depth,
    RWTexture2D<float4> image
) {
    uint width, height;
    image.GetDimensions(width, height);

    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // ------------------------------------------------------------
    // 1. Generate the camera ray in world space.
    // ------------------------------------------------------------

    float aspect = float(width) / float(height);

    float fov_x_rad = radians(camera.fov_x);
    float plane_w = 2.0 * tan(fov_x_rad * 0.5);
    float plane_h = plane_w / aspect;

    float u = (float(tid.x) + 0.5) / float(width);
    float v = (float(tid.y) + 0.5) / float(height);

    float xp = plane_w * (u - 0.5);
    float yp = plane_h * (0.5 - v);

    float3 ray_origin = xform_pt(
        camera.cam_to_world,
        float3(0.0, 0.0, 0.0)
    );

    float3 ray_dir = normalize(
        xform_vec(camera.cam_to_world, float3(xp, yp, -1.0))
    );

    float3 color = float3(0.0);
    float3 throughput = float3(1.0);

    // ------------------------------------------------------------
    // 2. Trace the ray, including mirror reflection bounces.
    // ------------------------------------------------------------

    for (int bounce = 0; bounce < max_depth; bounce++) {

        float t_min = 1e30;
        int hit_index = -1;
        float3 hit_normal_obj = float3(0.0);

        // Find the closest intersection with any shape.
        for (int i = 0; i < num_shapes; i++) {

            Shape sh = shapes[i];

            // Transform the ray from world space into object space.
            float3 obj_origin = xform_pt(
                sh.world_to_obj, ray_origin
            );

            float3 obj_dir = xform_vec(
                sh.world_to_obj, ray_dir
            );

            float t;
            float3 normal_obj;
            bool hit = false;

            // ----------------------------------------------------
            // Sphere intersection (same method as 3.4).
            // ----------------------------------------------------

            if (sh.type == ShapeType.Sphere) {

                Sphere sp = spheres[sh.index];

                float3 oc = obj_origin - sp.center;

                float A = dot(obj_dir, obj_dir);
                float B = 2.0 * dot(obj_dir, oc);
                float C = dot(oc, oc) - sp.radius * sp.radius;

                float2 roots;

                if (solve_quadratic(A, B, C, roots)) {

                    float t0 = roots.x;
                    float t1 = roots.y;

                    t = -1.0;

                    if (t0 > EPSILON && t1 > EPSILON) {
                        t = min(t0, t1);
                    } else if (t0 > EPSILON) {
                        t = t0;
                    } else if (t1 > EPSILON) {
                        t = t1;
                    }

                    if (t > EPSILON && t < t_min) {

                        normal_obj = normalize(
                            (obj_origin + t * obj_dir - sp.center)
                            / sp.radius
                        );

                        hit = true;
                    }
                }
            }

            // ----------------------------------------------------
            // Triangle mesh intersection (new in 3.5).
            // ----------------------------------------------------

            else if (sh.type == ShapeType.TriangleMesh) {

                TriangleMesh mesh = meshes[sh.index];

                hit = intersect_mesh(
                    mesh,
                    vertices,
                    vertex_normals,
                    triangles,
                    obj_origin,
                    obj_dir,
                    t_min,
                    t,
                    normal_obj
                );
            }

            // Update the closest hit across all shapes.
            if (hit && t < t_min) {
                t_min = t;
                hit_index = i;
                hit_normal_obj = normal_obj;
            }
        }

        // --------------------------------------------------------
        // 3. If no shape was hit, add the background and stop.
        // --------------------------------------------------------

        if (hit_index < 0) {
            color += throughput * background;
            break;
        }

        Shape hit_shape = shapes[hit_index];

        // Compute the hit point in world space.
        float3 hit_pos = ray_origin + t_min * ray_dir;

        // Transform the object-space normal into world space.
        // Normals use the inverse transpose transformation.
        float3x3 normal_matrix =
            transpose((float3x3)hit_shape.world_to_obj);

        float3 normal = normalize(
            mul(normal_matrix, hit_normal_obj)
        );

        // --------------------------------------------------------
        // 4. Direct lighting and shadow rays.
        // --------------------------------------------------------

        float3 direct_light = float3(0.0);

        for (int l = 0; l < num_lights; l++) {

            PointLight light = lights[l];

            float3 to_light = light.position - hit_pos;
            float dist_light = length(to_light);
            float3 l_dir = to_light / dist_light;

            bool in_shadow = false;

            // Check whether another shape blocks the light.
            for (int s = 0; s < num_shapes; s++) {

                Shape sh = shapes[s];

                float3 shadow_origin = xform_pt(
                    sh.world_to_obj, hit_pos
                );

                float3 shadow_dir = xform_vec(
                    sh.world_to_obj, l_dir
                );

                // Sphere shadow intersection.
                if (sh.type == ShapeType.Sphere) {

                    Sphere sp = spheres[sh.index];

                    float3 oc = shadow_origin - sp.center;

                    float A = dot(shadow_dir, shadow_dir);
                    float B = 2.0 * dot(shadow_dir, oc);
                    float C = dot(oc, oc) - sp.radius * sp.radius;

                    float2 roots;

                    if (solve_quadratic(A, B, C, roots)) {

                        if ((roots.x > EPSILON &&
                             roots.x < dist_light) ||
                            (roots.y > EPSILON &&
                             roots.y < dist_light)) {

                            in_shadow = true;
                            break;
                        }
                    }
                }

                // Triangle mesh shadow intersection.
                else if (sh.type == ShapeType.TriangleMesh) {

                    TriangleMesh mesh = meshes[sh.index];

                    float shadow_t;
                    float3 shadow_normal;

                    if (intersect_mesh(
                        mesh,
                        vertices,
                        vertex_normals,
                        triangles,
                        shadow_origin,
                        shadow_dir,
                        dist_light,
                        shadow_t,
                        shadow_normal
                    )) {
                        in_shadow = true;
                        break;
                    }
                }
            }

            // Add diffuse lighting if the light is not blocked.
            if (!in_shadow) {

                float n_dot_l = max(0.0, dot(normal, l_dir));

                float3 intensity =
                    light.intensity / (dist_light * dist_light);

                direct_light +=
                    hit_shape.Kd * intensity * n_dot_l;
            }
        }

        // Accumulate the lighting from this bounce.
        color += throughput * direct_light;

        // --------------------------------------------------------
        // 5. Mirror reflection.
        // --------------------------------------------------------

        if (max(
            hit_shape.Kr.r,
            max(hit_shape.Kr.g, hit_shape.Kr.b)
        ) <= 0.0) {
            break;
        }

        throughput *= hit_shape.Kr;

        ray_origin = hit_pos;
        ray_dir = reflect(ray_dir, normal);
    }

    image[tid] = float4(color, 1.0);
}


