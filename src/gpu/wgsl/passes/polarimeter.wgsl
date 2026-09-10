@group(0) @binding(0) var intensity_image: texture_2d<f32>;
@group(0) @binding(1) var q_image: texture_2d<f32>;
@group(0) @binding(2) var u_image: texture_2d<f32>;
@group(0) @binding(3) var output_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> measurement: vec4f;

/** Analyze already accumulated Stokes radiance; no ray or source evaluation occurs here. */
@compute @workgroup_size(8, 8)
fn measure(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(intensity_image))) { return; }
  let pixel = vec2i(id.xy);
  let intensity = textureLoad(intensity_image, pixel, 0);
  let q = textureLoad(q_image, pixel, 0).rgb;
  let u = textureLoad(u_image, pixel, 0).rgb;
  var color = intensity.rgb;
  if (measurement.z == 1) {
    color = 0.5 * (color + measurement.x * q + measurement.y * u);
  } else {
    let luminance = vec3f(0.2126, 0.7152, 0.0722);
    let total = dot(luminance, color);
    let linear = vec2f(dot(luminance, q), dot(luminance, u));
    color = vec3f(0);
    if (total > 0) {
      let degree = length(linear) / total;
      if (measurement.z == 2) { color = vec3f(degree); }
      else if (degree > 0) {
        let phase = atan2(linear.y, linear.x);
        color = degree * (0.5 + 0.5 * cos(phase + vec3f(0, -2.0943951, 2.0943951)));
      }
    }
  }
  textureStore(output_image, pixel, vec4f(color, intensity.a));
}
