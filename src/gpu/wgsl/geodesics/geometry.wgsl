/** Cartesian Kerr–Schild geometry. chart is +1 ingoing or −1 outgoing; Σ must be positive. */
struct KerrGeometry {
  position: vec3f,
  direction: vec3f,
  polar: vec3f,
  azimuthal: vec3f,
  radius: f32,
  spin: f32,
  sigma: f32,
  factor: f32,
  chart: f32,
}

fn kerr_geometry(space: vec2f, point: vec3f, chart: f32) -> KerrGeometry {
  let r = point.x;
  let a = space.x;
  var sine = sin(point.y);
  var cosine = cos(point.y);
  if (point.y == 0 || point.y == 3.141592653589793) { sine = 0; }
  if (point.y == 1.570796326794897) { cosine = 0; }
  let cp = cos(point.z);
  let sp = sin(point.z);
  let n = vec3f(sine * cp, sine * sp, cosine);
  let sigma = r * r + a * a * cosine * cosine;
  return KerrGeometry(
    r * n + chart * a * cross(vec3f(0, 0, 1), n), n,
    vec3f(cosine * cp, cosine * sp, -sine), vec3f(-sp, cp, 0),
    r, a, sigma, (2 * r - space.y * space.y) / sigma, chart,
  );
}

fn kerr_lower(g: KerrGeometry, tangent: vec4f) -> vec4f {
  let k = vec4f(1, g.chart * g.direction);
  return vec4f(-tangent.x, tangent.yzw) + g.factor * dot(k, tangent) * k;
}

fn kerr_raise(g: KerrGeometry, covector: vec4f) -> vec4f {
  let k = vec4f(-1, g.chart * g.direction);
  return vec4f(-covector.x, covector.yzw) - g.factor * dot(k, covector) * k;
}

/** Horizon-regular accelerated reference tetrad, columns (u, er, eθ, eφ). */
fn kerr_frame(g: KerrGeometry) -> mat4x4f {
  let f = g.factor;
  return mat4x4f(
    vec4f(1 + f / 2, -f * g.chart * g.direction / 2),
    vec4f(g.chart * f / 2, (1 - f / 2) * g.direction),
    vec4f(0, g.polar), vec4f(0, g.azimuthal),
  );
}

/** Lorentz boost of a tetrad by a local velocity with |β|² < 1. */
fn kerr_boost(frame: mat4x4f, beta: vec3f) -> mat4x4f {
  let gamma = inverseSqrt(1 - dot(beta, beta));
  let spatial = frame * vec4f(0, beta);
  let shift = gamma * frame[0] + gamma * gamma / (gamma + 1) * spatial;
  return mat4x4f(gamma * (frame[0] + spatial),
    frame[1] + beta.x * shift, frame[2] + beta.y * shift, frame[3] + beta.z * shift);
}

/** Principal-tensor contractions (h(f,p), *h(f,p)); no energy normalization or horizon pole. */
fn walker_penrose(g: KerrGeometry, p: vec4f, f: vec4f) -> vec2f {
  let x = g.position;
  let h = vec4f(dot(x, p.yzw), -x.x * p.x + g.spin * p.z,
    -x.y * p.x - g.spin * p.y, -x.z * p.x);
  let dual = vec4f(g.spin * p.w, -x.z * p.z + x.y * p.w,
    x.z * p.y - x.x * p.w, -g.spin * p.x - x.y * p.y + x.x * p.z);
  return vec2f(dot(h, f), dot(dual, f));
}

/** E, Lz, C and future radial/polar Mino derivatives for a neutral tangent. */
struct KerrMotion {
  constants: vec4f,
  radial: f32,
  angular: vec3f,
}

fn kerr_motion(g: KerrGeometry, tangent: vec4f, mass_squared: f32) -> KerrMotion {
  let p = kerr_lower(g, tangent);
  let sine = -g.polar.z;
  let cosine = g.direction.z;
  let n_momentum = dot(g.direction, p.yzw);
  let theta_momentum = dot(g.polar, p.yzw);
  let phi_momentum = dot(g.azimuthal, p.yzw);
  let polar = g.radius * theta_momentum + g.chart * g.spin * cosine * phi_momentum;
  let scaled_l = g.radius * phi_momentum - g.chart * g.spin * (sine * n_momentum + cosine * theta_momentum);
  let carter = polar * polar + cosine * cosine * (g.spin * g.spin * (mass_squared - p.x * p.x) + scaled_l * scaled_l);
  let radial = g.sigma * dot(g.direction, tangent.yzw) + g.chart * g.spin * sine *
    (-g.chart * g.spin * cosine * dot(g.polar, tangent.yzw) + g.radius * dot(g.azimuthal, tangent.yzw));
  return KerrMotion(vec4f(-p.x, sine * scaled_l, carter, mass_squared), radial,
    polar * g.azimuthal - scaled_l * g.polar);
}
