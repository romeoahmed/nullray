import { test } from "vitest";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync, mkdirSync } from "node:fs";
import { cpus, release } from "node:os";
import { createScene, initialScene } from "../src/scene/scene.ts";
import type { Scene, SceneInput } from "../src/scene/scene.ts";
import { createDiskProfile } from "../src/physics/disk.ts";
import { createStars, decodeFaintStars } from "../src/physics/sky.ts";
import { createStarTree } from "../src/physics/stars.ts";
import { createBlackbodyTable } from "../src/physics/radiation.ts";
import { turnCamera } from "../src/scene/camera.ts";

/** Keep persisted benchmark inputs independent of derived caches and typed-array layout. */
function sceneInputs(scene: Scene): SceneInput {
  const { space, observer, camera, disk, jet, plasma } = scene;
  return { space, observer, camera, disk, jet, plasma };
}
function prepare(input: SceneInput, previous?: Scene): Scene {
  const result = createScene(input, previous);
  if (!result.ok) {
    throw new Error(result.error);
  }
  return result.value;
}
const falling = prepare({
  ...initialScene,
  observer: {
    ...initialScene.observer,
    motion: { kind: "freefall", velocity: [0, 0, 0], properTime: 40 },
  },
});
const bytes = readFileSync(new URL("../src/data/faint-stars.bin", import.meta.url));
const catalogue = [...createStars(), ...decodeFaintStars(Uint8Array.from(bytes).buffer)];
const workloads = [
  {
    name: "scene",
    inputs: { previous: sceneInputs(initialScene), observerAzimuth: 0.1 },
    scope: "Change observer azimuth; reuse unchanged source profile.",
    run: () =>
      prepare(
        { ...initialScene, observer: { ...initialScene.observer, azimuth: 0.1 } },
        initialScene,
      ),
  },
  {
    name: "freefall-camera",
    inputs: { previous: sceneInputs(falling), cameraTurn: [0.1, 0.2], fieldOfView: 0.5 },
    scope: "Turn the camera and widen the field of view at an already prepared free-fall event.",
    run: () =>
      prepare(
        {
          ...falling,
          camera: turnCamera(falling.camera, 0.1, 0.2),
          observer: { ...falling.observer, fieldOfView: 0.5 },
        },
        falling,
      ),
  },
  {
    name: "disk",
    inputs: { space: initialScene.space, disk: initialScene.disk },
    scope: "Integrate the neutral circular-emitter profile and normalize its temperature.",
    run: () =>
      createDiskProfile(initialScene.space, initialScene.disk.inner, initialScene.disk.outer)
        .temperature,
  },
  {
    name: "stars",
    inputs: { sources: catalogue.length },
    scope: "Build the complete production stellar tree; catalogue decoding is outside timing.",
    run: () => createStarTree(catalogue),
  },
  {
    name: "thermal",
    inputs: { temperatureKelvin: [100, 1_000_000] },
    scope: "Integrate CIE spectra for the production temperature lookup.",
    run: () => createBlackbodyTable().data,
  },
];

test.for(workloads)("$name preparation", async (workload, { bench, annotate }) => {
  let result: Scene | Float32Array | undefined;
  await bench(
    `${workload.name} / wall-clock ms`,
    {
      writeResult: `test-results/bench/cpu-${workload.name}.json`,
    },
    () => {
      result = workload.run();
    },
  ).run({ iterations: 64, time: 1000, warmupIterations: 16, warmupTime: 250 });
  if (!result) {
    throw new Error("Preparation produced no result.");
  }
  // Consume completed work after timing without serializing megabytes of implementation arrays.
  const output =
    result instanceof Float32Array
      ? new Uint8Array(result.buffer, result.byteOffset, result.byteLength)
      : JSON.stringify({
          inputs: sceneInputs(result),
          event: result.prepared.geometry.point,
          velocity: result.prepared.frame.velocity,
        });
  const source = createHash("sha256");
  for (const directory of ["scene", "physics", "data"]) {
    const url = new URL(`../src/${directory}/`, import.meta.url);
    for (const file of readdirSync(url)
      .filter((name) => /\.(ts|bin)$/.test(name))
      .toSorted()) {
      source.update(JSON.stringify(`${directory}/${file}`));
      source.update(readFileSync(new URL(file, url)));
    }
  }
  const harness = createHash("sha256");
  for (const file of [
    "benchmarks/cpu.bench.ts",
    "vitest.config.ts",
    "package.json",
    "pnpm-lock.yaml",
  ]) {
    harness.update(readFileSync(file));
  }
  const body = JSON.stringify(
    {
      schema: 1,
      recordedAt: new Date().toISOString(),
      timing: { clock: "wall-clock", unit: "ms", completion: "synchronous" },
      workload: workload.name,
      inputs: workload.inputs,
      outputSHA256: createHash("sha256").update(output).digest("hex"),
      sourceSHA256: source.digest("hex"),
      harnessSHA256: harness.digest("hex"),
      runtime: process.version,
      cpu: cpus()[0]?.model,
      platform: process.platform,
      architecture: process.arch,
      release: release(),
      sampling: {
        minimumIterations: 64,
        minimumTimeMs: 1000,
        warmupIterations: 16,
        warmupTimeMs: 250,
      },
      scope: workload.scope,
    },
    null,
    2,
  );
  mkdirSync("test-results/bench", { recursive: true });
  writeFileSync(`test-results/bench/cpu-${workload.name}-context.json`, body);
  await annotate("Preparation context", {
    body,
    bodyEncoding: "utf-8",
    contentType: "application/json",
  });
});
