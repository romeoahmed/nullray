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

/**
 * Circular KS emitter with coordinate angular velocity Ω and stationary side ±1.
 * Returns zero if the circular worldline is not timelike.
 */
fn kerr_circular_velocity(g: KerrGeometry, omega: f32, side: i32) -> vec4f {
  let trial = vec4f(1, omega * cross(vec3f(0, 0, 1), g.position));
  let norm = -dot(kerr_lower(g, trial), trial);
  if (!(norm > 0)) { return vec4f(0); }
  return (f32(side) * inverseSqrt(norm)) * trial;
}

/**
 * Return (h(f,p), *h(f,p)) with signature −+++ and orientation ε(T,X,Y,Z)=+1.
 * For a vacuum null p and parallel-transported screen f these contractions are conserved;
 * plasma transfer must not use them as vacuum polarization invariants.
 */
fn walker_penrose(g: KerrGeometry, p: vec4f, f: vec4f) -> vec2f {
  let x = g.position;
  let h = vec4f(dot(x, p.yzw), -x.x * p.x + g.spin * p.z,
    -x.y * p.x - g.spin * p.y, -x.z * p.x);
  let dual = vec4f(g.spin * p.w, -x.z * p.z + x.y * p.w,
    x.z * p.y - x.x * p.w, -g.spin * p.x - x.y * p.y + x.x * p.z);
  return vec2f(dot(h, f), dot(dual, f));
}

/**
 * Unreduced (E, L, C, mass²), future radial Mino rate, and canonical angular vector J.
 * J is pole-free auxiliary data, not conserved Euclidean angular momentum.
 */
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
