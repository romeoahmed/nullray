import { expect, test } from "vitest";
import geometrySource from "../../src/gpu/wgsl/geodesics/geometry.wgsl?raw";
import orbitSource from "../../src/gpu/wgsl/geodesics/orbit.wgsl?raw";
import barrierSource from "../../src/gpu/wgsl/geodesics/barriers.wgsl?raw";
import { computeReadback, readPixels, test as gpuTest } from "./compute.ts";
import { createOptics } from "../../src/gpu/optics/engine.ts";
import { createScene, initialScene } from "../../src/scene/scene.ts";
import { initialAppearance } from "../../src/scene/appearance.ts";

/** Exact integer in units of 2^-149 for a finite binary32 input. */
function dyadic(value: number): bigint {
  const bits = new DataView(new ArrayBuffer(4));
  bits.setFloat32(0, value);
  const word = bits.getUint32(0);
  const exponent = (word >>> 23) & 255;
  const fraction = BigInt(word & 0x7fffff);
  const magnitude = exponent === 0 ? fraction : (fraction + 0x800000n) << BigInt(exponent - 1);
  return word >>> 31 ? -magnitude : magnitude;
}

/** Exact sign after clearing the positive plasma denominator; no floating point cancellation. */
function potentialSign(values: readonly number[]): bigint {
  const [a = 0n, q = 0n, e = 0n, l = 0n, c = 0n, r = 0n, amplitude = 0n, scale = 0n] =
    values.map(dyadic);
  const unit = 1n << 149n;
  const r2 = r * r;
  const p = e * (r2 + a * a) - a * l * unit;
  const delta = r2 - 2n * r * unit + a * a + q * q;
  const difference = l * unit - a * e;
  const k = difference * difference + c * unit ** 3n;
  const denominator = r2 + scale * unit;
  return (p * p - delta * k) * denominator - delta * amplitude * r2 * unit ** 3n;
}

test("radial source-exclusion signs agree with exact dyadic arithmetic", async () => {
  const cases: number[][] = [];
  for (const e of [-1, 0, 1]) {
    for (const radius of [0, 1, 3, 4, 10]) {
      for (const amplitude of [0, 20]) {
        cases.push([0.7, 0.2, e, 0, 30, radius, amplitude, 4]);
      }
    }
  }
  // Exact cancellation and tiny perturbations must never become an invented forbidden region.
  for (const c of [0, 2 ** -149, -(2 ** -149), 2 ** -22, -(2 ** -22)]) {
    cases.push([0.7, 0, 1, 0.2, c, 0, 0, 1]);
  }
  cases.push([1e30, 0, 1e30, 0, 0, 1, 0, 1]);
  const inputs = new Float32Array(cases.flat());
  const output = await computeReadback(
    `${geometrySource}\n${orbitSource}\n${barrierSource}
    @group(0) @binding(0) var<storage, read> input: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let first = input[id.x * 2]; let second = input[id.x * 2 + 1];
      var path: KerrOrbit;
      path.space = first.xy;
      path.constants = vec4f(first.zw, second.x, 0);
      path.plasma = second.zw;
      output[id.x] = select(0.0, 1.0, radial_forbidden(path, second.y));
    }`,
    inputs,
    cases.length,
    cases.length,
  );
  let certified = 0;
  for (let i = 0; i < cases.length; i++) {
    if (output[i] === 1) {
      certified++;
      expect(potentialSign(Array.from(inputs.slice(i * 8, i * 8 + 8))), `case ${i}`).toBeLessThan(
        0n,
      );
    }
  }
  expect(certified).toBeGreaterThan(10);
  expect(output[30]).toBe(0);
  expect(output.at(-1)).toBe(0);
});

gpuTest(
  "interior image and selected rays distinguish excluded sources from disk hits",
  async ({ device }) => {
    using owned = new DisposableStack();
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const scene = createScene({
      ...initialScene,
      space: { spin: 0.7, charge: 0.2 },
      disk: { inner: 3.304466962814331, outer: 64 },
      observer: {
        ...initialScene.observer,
        radius: 1,
        inclination: 1.2,
        azimuth: 0,
        fieldOfView: Math.PI / 3,
        motion: { kind: "regular" },
      },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const encoder = device.createCommandEncoder();
    const image = optics.encode(encoder, scene.value, 64, 36, {
      // This event fixture is the opaque surface limit, independent of the default volume opacity.
      appearance: { ...initialAppearance, diskThickness: 0 },
      time: 0,
      view: "image",
      jitter: [0, 0],
    });
    device.queue.submit([encoder.finish()]);
    const pixels = await readPixels(device, image.radiance);
    for (let i = 3; i < pixels.length; i += 4) {
      expect(pixels[i]).toBe(1);
    }
    // Pixel (54,0), minimized from the formerly unresolved interior image at 64×36.
    const trapped = await optics.inspect([54.5 / 64, 0.5 / 36]);
    expect(trapped.kind).toBe("source-free");
    expect(trapped.points.length).toBeGreaterThan(10);
    const disk = await optics.inspect([29.5 / 64, 8.5 / 36]);
    expect(disk.kind).toBe("disk");
    const horizon = createScene({
      ...initialScene,
      space: { spin: 0, charge: 0 },
      disk: { inner: 6, outer: 64 },
      observer: {
        ...initialScene.observer,
        radius: 2,
        inclination: 0,
        motion: { kind: "regular" },
      },
    });
    if (!horizon.ok) {
      throw new Error(horizon.error);
    }
    const generatorEncoder = device.createCommandEncoder();
    const generatorImage = optics.encode(generatorEncoder, horizon.value, 1, 1, {
      appearance: initialAppearance,
      time: 0,
      view: "image",
      jitter: [0, 0],
    });
    device.queue.submit([generatorEncoder.finish()]);
    expect(Array.from(await readPixels(device, generatorImage.radiance))).toEqual([0, 0, 0, 1]);
    const generator = await optics.inspect();
    expect(generator.kind).toBe("source-free");
    expect(generator.points[0]?.radius).toBe(2);
    const negative = createScene({
      ...horizon.value,
      observer: { ...horizon.value.observer, radius: -8 },
      camera: { forward: [1, 0, 0], up: [0, 1, 0] },
    });
    if (!negative.ok) {
      throw new Error(negative.error);
    }
    const negativeEncoder = device.createCommandEncoder();
    const negativeImage = optics.encode(negativeEncoder, negative.value, 1, 1, {
      appearance: initialAppearance,
      time: 0,
      view: "image",
      jitter: [0, 0],
    });
    device.queue.submit([negativeEncoder.finish()]);
    const negativePixels = await readPixels(device, negativeImage.radiance);
    expect(negativePixels[3]).toBe(1);
    const negativePath = await optics.inspect();
    expect(negativePath.kind).toBe("infinity");
    expect(negativePath.points.at(-1)?.block).toBe("disconnected");
    expect(negativePath.points.at(-1)?.radius).toBe(-Infinity);
  },
);

test("a zero-energy GPU orbit traverses both bifurcation spheres into the next universe", async () => {
  const result = await computeReadback(
    `${geometrySource}\n${orbitSource}
    @group(0) @binding(0) var<storage, read> input: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
    @compute @workgroup_size(1) fn main() {
      var path: KerrOrbit;
      path.space = input[0].xy;
      path.constants = vec4f(0, 0, 1, 0);
      path.plasma = vec2f(0, 1);
      path.chart = -1;
      path.state.radial = vec4f(1, sqrt(1 - dot(path.space, path.space)), 0, 0);
      path.state.direction = vec3f(1, 0, 0);
      path.state.angular = vec3f(0, 1, 0);
      var block = vec3i(3, 0, 0);
      let h = 6.283185307179586 / 1024;
      for (var i = 0u; i < 1024u; i++) {
        path = kerr_condition(path);
        let step = kerr_step(path, h);
        if (!step.valid) { output[0] = vec4f(-100); return; }
        let turn = path.bifurcation.z != 0 && sign(path.state.radial.y) != sign(step.state.radial.y);
        block = kerr_continue_block(path, step.state, h, block);
        path.state = step.state;
        if (turn) { path = kerr_bifurcation_flip(path); }
      }
      path = kerr_regular(path);
      output[0] = path.state.radial;
      output[1] = vec4f(path.state.direction, 0);
      output[2] = vec4f(vec3f(block), 0);
    }`,
    new Float32Array([0.5, 0.25, 0, 0]),
    12,
    1,
  );
  expect(result[0]).toBeCloseTo(1, 4);
  expect(result[1]).toBeCloseTo(Math.sqrt(0.6875), 4);
  expect(result[2]).toBeCloseTo(0, 4);
  expect(result[4]).toBeCloseTo(1, 4);
  expect(result[5]).toBeCloseTo(0, 4);
  expect(result[6]).toBeCloseTo(0, 4);
  expect(Array.from(result.slice(8, 11))).toEqual([3, 1, 0]);
});
