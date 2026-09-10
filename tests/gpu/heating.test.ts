import { expect, test } from "vitest";
import { server } from "vitest/browser";
import geometrySource from "../../src/gpu/wgsl/geodesics/geometry.wgsl?raw";
import orbitSource from "../../src/gpu/wgsl/geodesics/orbit.wgsl?raw";
import mediumSource from "../../src/gpu/wgsl/geodesics/medium.wgsl?raw";
import { kerrGeometry } from "../../src/physics/geometry.ts";
import { heatingProfile, plasmaProfile } from "../../src/physics/plasma.ts";
import { initialPlasma } from "../../src/scene/plasma.ts";
import { initialAppearance } from "../../src/scene/appearance.ts";
import { createScene, initialScene } from "../../src/scene/scene.ts";
import { createOptics } from "../../src/gpu/optics/engine.ts";
import { computeReadback, readPixels, test as gpuTest } from "./compute.ts";

test("heated-plasma GPU gradients match the physical scalar field", async () => {
  const space = { spin: Math.fround(0.7), charge: Math.fround(0.2) };
  const plasma = { ...initialPlasma, heating: 4 };
  const profile = plasmaProfile(plasma);
  const heat = heatingProfile(plasma);
  if (!heat) {
    throw new Error("Heated profile required.");
  }
  await Promise.all(
    (["ingoing", "outgoing"] as const).map(async (chart) => {
      const point = { radius: 16, inclination: Math.fround(1.1), azimuth: Math.fround(0.3), chart };
      const g = kerrGeometry(space, point);
      if (!g) {
        throw new Error("Regular medium fixture required.");
      }
      const sign = chart === "ingoing" ? 1 : -1;
      const scalar = (event: readonly number[]) => {
        const [time = 0, x = 0, y = 0, z = 0] = event;
        const a = space.spin;
        const rho = x * x + y * y + z * z - a * a;
        const r = Math.sqrt((rho + Math.sqrt(rho * rho + 4 * a * a * z * z)) / 2);
        const phi = Math.atan2(r * y - sign * a * x, r * x + sign * a * y);
        const nz = z / r;
        const delta = r * r - 2 * r + a * a + space.charge * space.charge;
        const d = Math.sqrt(1 - a * a - space.charge * space.charge);
        const primitive = Math.log((r - 1 - d) / (r - 1 + d)) / (2 * d);
        const ingoingPhi = phi + (1 - sign) * a * primitive;
        const ingoingTime =
          time +
          sign * r +
          (1 - sign) * (r + Math.log(delta) + (2 - space.charge * space.charge) * primitive);
        const coordinate = (r - heat.inner) / (heat.outer - heat.inner);
        const envelope = 64 * coordinate ** 3 * (1 - coordinate) ** 3;
        const wave =
          0.5 +
          0.5 *
            (1 - nz * nz) ** 3 *
            Math.cos(6 * ingoingPhi - heat.frequency * ingoingTime + 2 * Math.PI * coordinate);
        return (
          (profile.amplitude * r * r) /
          ((r * r + profile.scaleSquared) * (r * r + a * a * nz * nz)) /
          (1 + heat.contrast * envelope * wave)
        );
      };
      const event = [12 - sign * point.radius, ...g.position];
      const difference = 0.0001;
      const gradient = event.map((_, index) => {
        const right = event.map((value, i) => value + (i === index ? difference : 0));
        const left = event.map((value, i) => value - (i === index ? difference : 0));
        return (scalar(right) - scalar(left)) / (2 * difference);
      });
      const reference = { cutoff: scalar(event), gradient };
      const result = await computeReadback(
        `${geometrySource}\n${orbitSource}\n${mediumSource}
      @group(0) @binding(0) var<storage, read> input: array<vec4f>;
      @group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
      @compute @workgroup_size(1) fn main() {
        let g = kerr_geometry(input[0].xy, vec3f(input[0].zw, input[1].x), input[1].y);
        let geometry = medium_geometry(input[0].xy, g.radius, g.direction, g.chart);
        let medium = medium_potential(input[0].xy, geometry, input[2].xy, vec4f(input[2].zw, input[3].xy), input[1].z);
        output[0] = vec4f(medium.cutoff, medium.gradient.xyz);
        output[1] = vec4f(medium.gradient.w, 0, 0, 0);
      }`,
        new Float32Array([
          space.spin,
          space.charge,
          point.radius,
          point.inclination,
          point.azimuth,
          chart === "ingoing" ? 1 : -1,
          12,
          0,
          profile.amplitude,
          profile.scaleSquared,
          heat.inner,
          heat.outer,
          heat.contrast,
          heat.frequency,
          0,
          0,
        ]),
        8,
        1,
      );
      for (const [index, expected] of [reference.cutoff, ...reference.gradient].entries()) {
        expect(Math.abs((result[index] ?? NaN) - expected)).toBeLessThan(
          2e-6 * Math.max(0.01, Math.abs(expected)),
        );
      }
    }),
  );
});

gpuTest(
  "heated plasma changes ray transfer with temperature and physical epoch",
  async ({ device }) => {
    using owned = new DisposableStack();
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const report: { heating: number; time: number; resolved: number; radiance: number }[] = [];
    const render = async (heating: number, time: number) => {
      const scene = createScene({ ...initialScene, plasma: { ...initialPlasma, heating } });
      if (!scene.ok) {
        throw new Error(scene.error);
      }
      const encoder = device.createCommandEncoder();
      const image = optics.encode(encoder, scene.value, 64, 36, {
        appearance: initialAppearance,
        time,
        view: "image",
        jitter: [0, 0],
      });
      device.queue.submit([encoder.finish()]);
      const pixels = await readPixels(device, image.radiance);
      let resolved = 0,
        radiance = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        expect(pixels[i]).toBe(pixels[i + 1]);
        expect(pixels[i]).toBe(pixels[i + 2]);
        resolved += pixels[i + 3] ?? 0;
        radiance += pixels[i] ?? 0;
      }
      expect(radiance).toBeGreaterThan(0);
      report.push({ heating, time, resolved, radiance });
      return pixels;
    };
    const cold = await render(0, 0);
    const warm = await render(4, 0);
    const later = await render(4, 50);
    expect(
      warm.some(
        (value, index) => index % 4 !== 3 && Math.abs(value - (cold[index] ?? NaN)) > 0.001,
      ),
    ).toBe(true);
    expect(
      later.some(
        (value, index) => index % 4 !== 3 && Math.abs(value - (warm[index] ?? NaN)) > 0.001,
      ),
    ).toBe(true);
    await server.commands.writeFile("test-results/heating.json", JSON.stringify(report, null, 2));
  },
);
