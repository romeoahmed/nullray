import { expect, test } from "vitest";
import geometrySource from "../../src/gpu/wgsl/geodesics/geometry.wgsl?raw";
import orbitSource from "../../src/gpu/wgsl/geodesics/orbit.wgsl?raw";
import { combine, kerrGeometry } from "../../src/physics/geometry.ts";
import type { FourVector } from "../../src/physics/geometry.ts";
import { plasmaCutoff } from "../../src/physics/plasma.ts";
import { referenceGeodesic } from "../reference/hamiltonian.ts";
import { boostFrame, observerPhoton, principalFrame } from "../../src/physics/observer.ts";
import { motionFromTangent } from "../../src/physics/motion.ts";
import { walkerPenrose } from "../reference/polarization.ts";
import { computeReadback } from "./compute.ts";
import { test as gpuTest, readPixels } from "./compute.ts";
import { createPolarimeter } from "../../src/gpu/imaging/polarimeter.ts";
import { createOptics } from "../../src/gpu/optics/engine.ts";
import { createScene, initialScene } from "../../src/scene/scene.ts";
import { initialAppearance } from "../../src/scene/appearance.ts";
import { initialJet } from "../../src/scene/jet.ts";
import { initialPlasma } from "../../src/scene/plasma.ts";
import emptyTail from "../regressions/material-empty-tail.json";
import horizonReflection from "../regressions/inner-horizon-reflection.json";

gpuTest(
  "near-inner-horizon rays retain their causal source domain and a dark shadow",
  async ({ device }) => {
    using owned = new DisposableStack();
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const fixture = horizonReflection;
    const scene = createScene({
      space: fixture.space,
      observer: fixture.observer,
      disk: fixture.disk,
      camera: {
        forward: [-1, 0, 0],
        up: [0, -Math.cos(fixture.cameraRoll), Math.sin(fixture.cameraRoll)],
      },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const encoder = device.createCommandEncoder();
    const image = optics.encode(encoder, scene.value, fixture.width, fixture.height, {
      appearance: {
        ...initialAppearance,
        diskTemperature: 30000,
        diskThickness: 0.045,
        diskOpticalDepth: 0.75,
        otherUniverses: 0,
        skyBrightness: 0,
      },
      time: fixture.time,
      view: "image",
      jitter: [0, 0],
    });
    device.queue.submit([encoder.finish()]);
    const pixels = await readPixels(device, image.radiance);
    for (const pixel of fixture.pixels) {
      const [x, y] = pixel;
      if (x === undefined || y === undefined) {
        throw new Error("Missing detector pixel.");
      }
      const offset = 4 * (y * fixture.width + x);
      expect(Array.from(pixels.slice(offset, offset + 4))).toEqual([0, 0, 0, 1]);
      // oxlint-disable-next-line no-await-in-loop
      const path = await optics.inspect([(x + 0.5) / fixture.width, (y + 0.5) / fixture.height]);
      expect(["infinity", "disk"]).toContain(path.kind);
      expect(path.points.some((point) => point.block === "interior")).toBe(true);
      expect(path.points.at(-1)?.universe).toBe(-1);
    }
  },
);

gpuTest(
  "material clipping retains foreground light before an empty asymptotic tail",
  async ({ device }) => {
    using owned = new DisposableStack();
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const scene = createScene({
      space: emptyTail.space,
      observer: emptyTail.observer,
      disk: emptyTail.disk,
      camera: { forward: [-1, 0, 0], up: [0, -Math.cos(-0.45), Math.sin(-0.45)] },
      jet: { ...initialJet, gammaMin: 1600, density: 2e6, openingAngle: 0.055 },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const encoder = device.createCommandEncoder();
    const image = optics.encode(encoder, scene.value, emptyTail.width, emptyTail.height, {
      appearance: {
        ...initialAppearance,
        diskTemperature: 30000,
        diskThickness: 0.06,
        diskOpticalDepth: 1.2,
        skyBrightness: 48,
      },
      time: emptyTail.time,
      view: "image",
      jitter: [0, 0],
    });
    device.queue.submit([encoder.finish()]);
    const pixels = await readPixels(device, image.radiance);
    for (const pixel of emptyTail.pixels) {
      const [x, y] = pixel;
      if (x === undefined || y === undefined) {
        throw new Error("Missing detector pixel.");
      }
      const offset = (y * emptyTail.width + x) * 4;
      expect(pixels[offset + 3]).toBe(1);
      expect(pixels[offset + 1]).toBeGreaterThan(0);
      // oxlint-disable-next-line no-await-in-loop
      const path = await optics.inspect([
        (x + 0.5) / emptyTail.width,
        (y + 0.5) / emptyTail.height,
      ]);
      expect(path.kind).toBe("infinity");
    }
  },
);

test("refracted GPU trajectories agree with an independent plasma Hamiltonian", async () => {
  const space = { spin: Math.fround(0.8), charge: Math.fround(0.3) };
  const inclination = Math.fround(0.9);
  const g = kerrGeometry(space, { radius: 8, inclination, azimuth: 0, chart: "ingoing" });
  if (!g) {
    throw new Error("Regular plasma fixture required.");
  }
  const plasma = { amplitude: 50, scaleSquared: 100 };
  const observer = principalFrame(g);
  const vacuum = observerPhoton(observer, [0.6, 0.48, 0.64]);
  const speed = Math.sqrt(1 - plasmaCutoff(plasma, g));
  const launch = combine(observer.velocity, 1 - speed, vacuum, speed);
  const p: FourVector = [
    Math.fround(launch[0]),
    Math.fround(launch[1]),
    Math.fround(launch[2]),
    Math.fround(launch[3]),
  ];
  const motion = motionFromTangent(space, g, p, 0);
  const reference = referenceGeodesic(
    space,
    {
      ...motion.constants,
      radialVelocity: motion.radialVelocity,
      polarVelocity: motion.polarVelocity,
    },
    8,
    inclination,
    0.008,
    1e-12,
    plasma,
  );
  const result = await computeReadback(
    `${geometrySource}\n${orbitSource}
    @group(0) @binding(0) var<storage, read> input: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
    @compute @workgroup_size(1) fn main() {
      let g = kerr_geometry(input[0].xy, vec3f(input[0].zw, 0), 1);
      var path = kerr_orbit(g, input[0].y, input[2], 0);
      path.plasma = input[1].xy;
      for (var i = 0u; i < 16u; i++) { path = kerr_condition(path); path.state = kerr_step(path, -0.0005).state; }
      var r = path.state.radial.x;
      if (path.inverse) { r = 1 / r; }
      let n = normalize(path.state.direction);
      let theta = acos(n.z);
      let phi = atan2(n.y, n.x);
      let start = kerr_chart_primitives(path.space, input[0].z);
      let end = kerr_chart_primitives(path.space, r);
      output[0] = vec4f(r, theta, phi - path.chart * end.x + start.x, path.state.radial.z - path.chart * end.y + start.y);
      let endpoint = kerr_geometry(path.space, vec3f(r, theta, phi), path.chart);
      let p = kerr_tangent(path);
      let cutoff = path.plasma.x * r * r / ((r * r + path.plasma.y) * endpoint.sigma);
      output[1] = vec4f(dot(kerr_lower(endpoint, p), p) + cutoff, 0, 0, 0);
    }`,
    new Float32Array([
      space.spin,
      space.charge,
      8,
      inclination,
      plasma.amplitude,
      plasma.scaleSquared,
      0,
      0,
      ...p,
    ]),
    8,
    1,
  );
  for (let i = 0; i < 4; i++) {
    const expected = reference.state[i] ?? NaN;
    expect(Math.abs((result[i] ?? NaN) - expected)).toBeLessThan(
      3e-5 * Math.max(1, Math.abs(expected)),
    );
  }
  expect(Math.abs(result[4] ?? NaN)).toBeLessThan(3e-5);
});

gpuTest("optically thin jet radiance scales with electron density", async ({ device }) => {
  using owned = new DisposableStack();
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const images: Float32Array[] = [];
  // Keep the material support fixed so differences isolate density from source-integration quadrature.
  for (const scale of [1, 2, 3]) {
    const scene = createScene({
      ...initialScene,
      observer: { ...initialScene.observer, radius: 50, inclination: 0.7, fieldOfView: 1.2 },
      jet: { ...initialJet, density: scale * initialJet.density },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const command = device.createCommandEncoder();
    const output = optics.encode(command, scene.value, 24, 24, {
      appearance: initialAppearance,
      time: 0,
      view: "image",
      jitter: [0, 0],
    });
    device.queue.submit([command.finish()]);
    // The next encode overwrites the same owned target.
    // oxlint-disable-next-line no-await-in-loop
    images.push(await readPixels(device, output.radiance));
  }
  const [background, single, double] = images;
  if (!background || !single || !double) {
    throw new Error("Missing jet images.");
  }
  expect(images.every((image) => image.every(Number.isFinite))).toBe(true);
  let flux = 0,
    doubledFlux = 0,
    resolved = 0;
  for (let i = 0; i < background.length; i += 4) {
    if (background[i + 3] !== 1 || single[i + 3] !== 1 || double[i + 3] !== 1) {
      continue;
    }
    resolved++;
    flux += (single[i + 1] ?? NaN) - (background[i + 1] ?? NaN);
    doubledFlux += (double[i + 1] ?? NaN) - (background[i + 1] ?? NaN);
  }
  expect(resolved).toBeGreaterThan((24 * 24) / 2);
  expect(flux).toBeGreaterThan(0);
  expect(Math.abs(doubledFlux / flux - 2)).toBeLessThan(0.02);
});

gpuTest("the production optical pass resolves finite disk and sky radiance", async ({ device }) => {
  using owned = new DisposableStack();
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const encoder = device.createCommandEncoder();
  const image = optics.encode(encoder, initialScene, 48, 32, {
    appearance: initialAppearance,
    time: 0,
    view: "image",
    jitter: [0, 0],
  });
  device.queue.submit([encoder.finish()]);
  const pixels = await readPixels(device, image.radiance);
  expect(pixels.every(Number.isFinite)).toBe(true);
  expect(pixels.some((value, index) => index % 4 < 3 && value > 0)).toBe(true);
  expect(
    pixels.filter((_, index) => index % 4 === 3).every((weight) => weight === 0 || weight === 1),
  ).toBe(true);
  const q = await readPixels(device, image.q);
  const u = await readPixels(device, image.u);
  expect(q.every(Number.isFinite) && u.every(Number.isFinite)).toBe(true);
  let polarized = false;
  for (let i = 0; i < pixels.length; i += 4) {
    const y = (data: Float32Array) =>
      0.2126 * (data[i] ?? 0) + 0.7152 * (data[i + 1] ?? 0) + 0.0722 * (data[i + 2] ?? 0);
    const linear = Math.hypot(y(q), y(u));
    polarized ||= linear > 1e-4;
    // The Milne source peaks at 11.713%; allow half-float storage rounding.
    expect(linear).toBeLessThanOrEqual(0.118 * Math.max(0, y(pixels)) + 1e-6);
  }
  expect(polarized).toBe(true);
  const detector = owned.adopt(await createPolarimeter(device), (value) => value.dispose());
  const measure = async (angle: number) => {
    const command = device.createCommandEncoder();
    const output = detector.encode(command, image, angle, "image");
    device.queue.submit([command.finish()]);
    return readPixels(device, output);
  };
  const measurements = [await measure(0), await measure(Math.PI / 2)];
  for (let i = 0; i < pixels.length; i++) {
    if (i % 4 === 3) {
      continue;
    }
    const intensity = pixels[i] ?? NaN;
    expect(
      Math.abs((measurements[0]?.[i] ?? NaN) + (measurements[1]?.[i] ?? NaN) - intensity),
    ).toBeLessThanOrEqual(0.002 * Math.max(1, Math.abs(intensity)));
  }
  const axial = createScene({
    space: { spin: 0.8, charge: 0.3 },
    observer: {
      radius: 3,
      inclination: 0,
      azimuth: 0,
      fieldOfView: 1,
      chart: "outgoing",
      motion: { kind: "regular" },
    },
  });
  if (!axial.ok) {
    throw new Error(axial.error);
  }
  const probe = device.createCommandEncoder();
  const ray = optics.encode(probe, axial.value, 1, 1, {
    appearance: initialAppearance,
    time: 0,
    view: "image",
    jitter: [0, 0],
  });
  const staging = owned.adopt(
    device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    (value) => value.destroy(),
  );
  probe.copyTextureToBuffer(
    { texture: ray.domains },
    { buffer: staging, bytesPerRow: 256 },
    [1, 1],
  );
  device.queue.submit([probe.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  // A backward principal ray traverses the white-hole and inner horizons to the negative-radius end.
  expect(Array.from(new Int32Array(staging.getMappedRange(), 0, 4))).toEqual([1, -1, 1, -1]);
  staging.unmap();
  const path = await optics.inspect();
  expect(path.kind).toBe("infinity");
  expect(path.points.at(-1)?.radius).toBe(-Infinity);
  expect(path.points.some((point) => point.block === "white-hole")).toBe(true);
  expect(path.points.at(-1)?.universe).toBe(-1);
  // Independent Schwarzschild boundary: sin(alpha_c)=3 sqrt(3) sqrt(1-2/R)/R for a static detector.
  for (const factor of [0.99, 1.01]) {
    const radius = 65;
    const alpha = Math.asin((factor * 3 * Math.sqrt(3) * Math.sqrt(1 - 2 / radius)) / radius);
    const critical = createScene({
      space: { spin: 0, charge: 0 },
      observer: { radius, inclination: 0, azimuth: 0, fieldOfView: 1, motion: { kind: "static" } },
      disk: { inner: 1000, outer: 1001 },
      camera: { forward: [-Math.cos(alpha), Math.sin(alpha), 0], up: [0, 0, 1] },
    });
    if (!critical.ok) {
      throw new Error(critical.error);
    }
    const command = device.createCommandEncoder();
    optics.encode(command, critical.value, 1, 1, {
      appearance: { ...initialAppearance, diskThickness: 0 },
      time: 0,
      view: "image",
      jitter: [0, 0],
    });
    device.queue.submit([command.finish()]);
    // Inspection must finish before the shared frame changes.
    // oxlint-disable-next-line no-await-in-loop
    const boundary = await optics.inspect();
    expect(boundary.kind).toBe(factor < 1 ? "singularity" : "infinity");
  }
  const horizonScene = createScene({
    space: { spin: 0, charge: 0 },
    observer: {
      radius: 2,
      inclination: 0,
      azimuth: 0,
      fieldOfView: 1,
      motion: { kind: "regular" },
    },
    camera: { forward: [1, 0, 0], up: [0, -1, 0] },
  });
  if (!horizonScene.ok) {
    throw new Error(horizonScene.error);
  }
  const horizonEncoder = device.createCommandEncoder();
  optics.encode(horizonEncoder, horizonScene.value, 1, 1, {
    appearance: initialAppearance,
    time: 0,
    view: "image",
    jitter: [0, 0],
  });
  device.queue.submit([horizonEncoder.finish()]);
  const horizonPath = await optics.inspect();
  expect(horizonPath.kind).toBe("infinity");
  expect(horizonPath.points.at(-1)?.block).toBe("exterior");
  expect(horizonPath.points.at(-1)?.radius).toBe(Infinity);
  const plasmaScene = createScene({ ...initialScene, plasma: initialPlasma });
  if (!plasmaScene.ok) {
    throw new Error(plasmaScene.error);
  }
  const plasmaEncoder = device.createCommandEncoder();
  const plasmaImage = optics.encode(plasmaEncoder, plasmaScene.value, 24, 24, {
    appearance: initialAppearance,
    time: 0,
    view: "image",
    jitter: [0, 0],
  });
  device.queue.submit([plasmaEncoder.finish()]);
  const narrowband = await readPixels(device, plasmaImage.radiance);
  expect(narrowband.every(Number.isFinite)).toBe(true);
  expect(narrowband.some((value, index) => index % 4 < 3 && value > 0)).toBe(true);
  const plasmaQ = await readPixels(device, plasmaImage.q);
  expect(plasmaQ.every((value, index) => index % 4 === 3 || value === 0)).toBe(true);
});

const quantize = (v: FourVector): FourVector => [
  Math.fround(v[0]),
  Math.fround(v[1]),
  Math.fround(v[2]),
  Math.fround(v[3]),
];

test("uploaded observer frames preserve native f32 Carter data and polarization across the extended geometry", async () => {
  const cases = [
    [0.8, 0.3, 8, 0.9, 0.3, 1],
    [0.8, 0.3, 1, 0.9, 0.3, 1],
    [0.8, 0.3, 0.1, 0.9, 0.3, -1],
    [0.8, 0.3, 0, 0, 0.3, 1],
    [0.8, 0.3, -2, 0.9, 0.3, 1],
    [1.2, 0.6, 1, 0.9, 0.3, -1],
  ].map((entry) => entry.map(Math.fround));
  const fixtures = cases.map(
    ([spin = NaN, charge = NaN, radius = NaN, inclination = NaN, azimuth = NaN, sign = NaN]) => {
      const space = { spin, charge };
      const g = kerrGeometry(space, {
        radius,
        inclination,
        azimuth,
        chart: sign > 0 ? "ingoing" : "outgoing",
      });
      const boosted =
        g && boostFrame(principalFrame(g), [Math.fround(0.2), Math.fround(-0.3), Math.fround(0.1)]);
      if (!g || !boosted) {
        throw new Error("Expected a regular timelike fixture.");
      }
      const frame = {
        velocity: quantize(boosted.velocity),
        radial: quantize(boosted.radial),
        polar: quantize(boosted.polar),
        azimuthal: quantize(boosted.azimuthal),
      };
      return {
        space,
        g,
        frame,
        data: [
          spin,
          charge,
          0,
          0,
          radius,
          inclination,
          azimuth,
          sign,
          ...frame.velocity,
          ...frame.radial,
          ...frame.polar,
          ...frame.azimuthal,
        ],
      };
    },
  );
  const source = `${geometrySource}
    struct Input { space: vec4f, point: vec4f, frame: mat4x4f }
    @group(0) @binding(0) var<storage, read> inputs: array<Input>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let input = inputs[id.x];
      let g = kerr_geometry(input.space.xy, input.point.xyz, input.point.w);
      let frame = input.frame;
      let p = frame * vec4f(1, -0.6, -0.48, -0.64);
      let motion = kerr_motion(g, p, 0);
      let wp = walker_penrose(g, p, frame[2]);
      outputs[id.x * 3] = motion.constants;
      outputs[id.x * 3 + 1] = vec4f(motion.radial, motion.angular);
      outputs[id.x * 3 + 2] = vec4f(wp, dot(kerr_lower(g, p), p), dot(kerr_lower(g, frame[0]), p));
    }`;
  const results = await computeReadback(
    source,
    new Float32Array(fixtures.flatMap(({ data }) => data)),
    cases.length * 12,
    cases.length,
  );
  for (const [i, { space, g, frame }] of fixtures.entries()) {
    const p = observerPhoton(frame, [Math.fround(0.6), Math.fround(0.48), Math.fround(0.64)]);
    const motion = motionFromTangent(space, g, p, 0);
    const wp = walkerPenrose(space, g, p, frame.polar);
    const expected = [
      motion.constants.energy,
      motion.constants.angularMomentum,
      motion.constants.carter,
      0,
      motion.radialVelocity,
      ...motion.angular,
      wp.real,
      wp.imaginary,
      0,
      -1,
    ];
    for (const [j, value] of expected.entries()) {
      expect(Math.abs((results[i * 12 + j] ?? NaN) - value)).toBeLessThan(
        3e-5 * Math.max(1, Math.abs(value)),
      );
    }
  }
});
