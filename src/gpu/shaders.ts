import composition from "./wgsl/imaging/stellar-raster.wgsl?raw";
import ownership from "./wgsl/imaging/stellar-partition.wgsl?raw";
import observer64 from "./wgsl/geodesics/observer64.wgsl?raw";
import dual64 from "./wgsl/math/dual64.wgsl?raw";
import dualLaunch64 from "./wgsl/geodesics/launch-dual64.wgsl?raw";
import dualElliptic64 from "./wgsl/math/cubic-dual64.wgsl?raw";
import dualQuartic64 from "./wgsl/math/quartic-dual64.wgsl?raw";
import dualSky64 from "./wgsl/lensing/sky-jacobian64.wgsl?raw";
import cooperative from "./wgsl/lensing/sky-integral.wgsl?raw";
import derivative from "./wgsl/math/dual32.wgsl?raw";
import differential from "./wgsl/math/quartic-dual32.wgsl?raw";
import imageSource from "./wgsl/lensing/sky-jacobian32.wgsl?raw";
import elliptic from "./wgsl/math/elliptic.wgsl?raw";
import arrival from "./wgsl/geodesics/arrival.wgsl?raw";
import equator from "./wgsl/geodesics/equator.wgsl?raw";
import transport from "./wgsl/geodesics/transport.wgsl?raw";
import radiation from "./wgsl/imaging/radiation.wgsl?raw";
import orbit from "./wgsl/geodesics/orbit.wgsl?raw";
import tracing from "./wgsl/geodesics/trace.wgsl?raw";
import soft64 from "./wgsl/math/binary64.wgsl?raw";
import launch64 from "./wgsl/geodesics/launch64.wgsl?raw";
import prepare64 from "./wgsl/math/quartic64.wgsl?raw";
import trace64 from "./wgsl/geodesics/trace64.wgsl?raw";
import beamSource from "./wgsl/lensing/beam.wgsl?raw";
import beamProjection from "./wgsl/lensing/beam-projection.wgsl?raw";
import edgeBeamSource from "./wgsl/passes/edge-beams.wgsl?raw";
import edgeSource from "./wgsl/passes/edges.wgsl?raw";
import repairSource from "./wgsl/passes/repair.wgsl?raw";
import geometrySource from "./wgsl/passes/geometry.wgsl?raw";
import diagnostic from "./wgsl/imaging/diagnostic.wgsl?raw";
import starsSource from "./wgsl/imaging/source-tree.wgsl?raw";
import stellarPatchSource from "./wgsl/imaging/stellar-patch.wgsl?raw";
import skySource from "./wgsl/passes/sky.wgsl?raw";

/** Exact shader modules used by the renderer and benchmark source fingerprints. */
const raySource = `${elliptic}\n${equator}\n${arrival}\n${transport}\n${orbit}\n${tracing}`;
const precisionSource = `${raySource}\n${soft64}\n${observer64}\n${launch64}\n${prepare64}\n${trace64}`;
const derivativeSource = `enable subgroups;\n${precisionSource}\n${derivative}\n${differential}\n${imageSource}\n${cooperative}\n${dual64}\n${dualLaunch64}\n${dualElliptic64}\n${dualQuartic64}\n${dualSky64}`;
export const opticalSources = {
  ray: raySource,
  precision: precisionSource,
  derivative: derivativeSource,
  geometry: `${precisionSource}\n${geometrySource}`,
  beam: `${derivativeSource}\n${beamSource}\n${beamProjection}\n${repairSource}`,
  edgeBeam: `${derivativeSource}\n${beamSource}\n${beamProjection}\n${repairSource}\n${edgeBeamSource}`,
  edge: `enable subgroups;\n${precisionSource}\n${edgeSource}`,
  sky: `${composition}\n${ownership}\n${radiation}\n${starsSource}\n${stellarPatchSource}\n${beamSource}\n${diagnostic}\n${skySource}`,
} as const;
