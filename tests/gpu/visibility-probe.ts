import { opticalSources } from "../../src/gpu/optics.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import type { Scene } from "../../src/scene/scene.ts";
import { computeReadback, observerData } from "./compute.ts";

/** Probe the native f32 classifier at prescribed local directions, independent of raster resolution. */
export async function traceDirections(
  directions: readonly Vec3[],
  inclination: number,
  scene?: Scene,
) {
  const frame = new Float32Array(24);
  frame.set([
    1,
    1,
    0,
    0,
    scene?.observer.radius ?? 30,
    inclination,
    1,
    scene?.observer.azimuth ?? 0,
    scene?.space.spin ?? 0,
    scene?.space.charge ?? 0,
    scene?.diskInner ?? 6,
    scene?.diskOuter ?? 64,
  ]);
  return computeReadback(
    `${opticalSources.ray}
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(2) var<storage, read> directions: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&directions)) { output[id.x] = trace_ray(frame, directions[id.x].xyz).value; }
}`,
    frame,
    directions.length * 4,
    Math.ceil(directions.length / 64),
    [new Float32Array(directions.flatMap((direction) => [...direction, 0]))],
  );
}

/** Production precision selection at explicit screen positions; each record is endpoint plus delay. */
export function tracePixels(frame: Float32Array, pixels: readonly (readonly [number, number])[]) {
  return computeReadback(
    `${opticalSources.precision}
@group(0) @binding(0) var<storage, read> frame: Frame;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(2) var<storage, read> pixels: array<vec2f>;
@group(0) @binding(3) var<storage, read> observer: Observer64;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&pixels)) { return; }
  let endpoint = screen_ray_refined(frame, pixels[id.x], observer);
  output[2u * id.x] = endpoint.value;
  output[2u * id.x + 1u] = vec4f(endpoint.relative_time, 0.0, 0.0, 0.0);
}`,
    frame,
    pixels.length * 8,
    Math.ceil(pixels.length / 64),
    [new Float32Array(pixels.flat()), new Float32Array(observerData(frame).buffer)],
  );
}
