#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float4 position [[attribute(0)]];
    float4 normal [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float3 edgeFactor;
};

vertex VertexOut wireframe_vertex(VertexIn in [[stage_in]],
                                constant float4x4 &mvp [[buffer(1)]]) {
    VertexOut out;
    out.position = mvp * in.position;
    out.normal = in.normal.xyz;
    out.edgeFactor = float3(1.0); // Edge detection factor
    return out;
}

fragment float4 wireframe_fragment(VertexOut in [[stage_in]],
                                 float3 barycentric [[barycentric_coord]]) {
    float3 deltas = fwidth(barycentric);
    float3 smoothing = deltas * 1.0;
    float3 thickness = deltas * 0.5;
    
    float3 wireframe = smoothstep(thickness, thickness + smoothing, barycentric);
    float factor = min(min(wireframe.x, wireframe.y), wireframe.z);
    
    return float4(1.0, 1.0, 1.0, 1.0 - factor); // White wireframe
}
