/** Host-shareable copies of differentiated paths; boolean flags become u32 only at storage boundaries. */
struct StoredStellarElliptic {
  values: array<vec3f, 6>,
  three_real: u32,
  jacobi: DifferentialJacobiPlan,
}
struct StoredStellarQuartic {
  values: array<vec3f, 7>,
  elliptic: StoredStellarElliptic,
}
struct StoredStellarPolar {
  values: array<vec3f, 7>,
  enabled: u32,
  jacobi: DifferentialJacobiPlan,
}
struct StoredStellarBeam {
  radial: StoredStellarQuartic,
  polar: StoredStellarQuartic,
  values: array<vec3f, 6>,
  motion: StoredStellarPolar,
  basis: vec4u,
  flags: vec4u,
}
/** 1456 bytes: prepared trial, accepted endpoint/step, parameter records, and bounded solver state. */
struct StellarImageWork {
  prepared: StoredStellarBeam,
  endpoint: vec4f,
  geometry: vec4f,
  previous: vec4f,
  parameter_words: array<vec4u,2>,
  trial_words: array<vec4u,2>,
  /** Status, accepted iterations, backtrack index, and presence of an accepted point. */
  state: vec4u,
}
@group(0) @binding(11) var<storage, read_write> stellar_work: array<StellarImageWork>;
@group(0) @binding(12) var<uniform> stellar_iteration: vec4u;

fn store_stellar_quartic(path: DifferentialQuartic) -> StoredStellarQuartic {
  let e = path.elliptic;
  return StoredStellarQuartic(
    array<vec3f,7>(path.x, path.velocity, path.f, path.first, path.second, path.third, path.fourth),
    StoredStellarElliptic(array<vec3f,6>(e.g2, e.g3, e.root, e.scale, e.m, e.complement),
      select(0u, 1u, e.three_real), e.jacobi));
}
fn load_stellar_quartic(stored: StoredStellarQuartic) -> DifferentialQuartic {
  let v = stored.values;
  let e = stored.elliptic.values;
  return DifferentialQuartic(v[0], v[1], v[2], v[3], v[4], v[5], v[6],
    DifferentialElliptic(e[0], e[1], e[2], e[3], e[4], e[5], stored.elliptic.three_real != 0u,
      stored.elliptic.jacobi));
}
fn store_stellar_beam(prepared: PreparedSkyBeam) -> StoredStellarBeam {
  let p = prepared.path;
  let m = p.motion;
  return StoredStellarBeam(store_stellar_quartic(p.radial), store_stellar_quartic(p.polar),
    array<vec3f,6>(p.energy, p.momentum, p.carter, p.end, p.mu, p.launch),
    StoredStellarPolar(array<vec3f,7>(m.phase, m.frequency, m.m, m.amplitude2, m.complement,
      m.signed_root, m.coefficient), select(0u, 1u, m.enabled), m.jacobi),
    prepared.basis, vec4u(select(0u, 1u, prepared.refined), select(0u, 1u, p.valid), 0u, 0u));
}
fn load_stellar_beam(stored: StoredStellarBeam) -> PreparedSkyBeam {
  let v = stored.values;
  let m = stored.motion.values;
  return PreparedSkyBeam(DifferentialSkyPath(load_stellar_quartic(stored.radial),
    load_stellar_quartic(stored.polar), v[0], v[1], v[2], v[3], v[4], v[5],
    DifferentialPolar(m[0], m[1], m[2], m[3], m[4], m[5], m[6], stored.motion.enabled != 0u,
      stored.motion.jacobi), stored.flags.y != 0u), stored.basis, stored.flags.x != 0u);
}

fn load_stellar_parameters(words: array<vec4u,2>) -> array<Soft64,3> {
  return array<Soft64,3>(words[0].xy,words[0].zw,words[1].xy);
}
fn store_stellar_parameters(values: array<Soft64,3>) -> array<vec4u,2> {
  return array<vec4u,2>(vec4u(values[0],values[1]),vec4u(values[2],vec2u(0u)));
}
