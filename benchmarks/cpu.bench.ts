import { expect, test } from "vitest";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync, mkdirSync } from "node:fs";
import { cpus, release } from "node:os";
import { createScene, initialScene } from "../src/scene/scene.ts";
import { createDiskProfile } from "../src/physics/disk.ts";
import { createSkyMap, createStars } from "../src/physics/sky.ts";
import { createStarTree } from "../src/physics/stars.ts";
import { createBlackbodyTable } from "../src/physics/radiation.ts";

const workloads = [
  {
    name: "scene",
    run: () => {
      const scene = createScene(
        { ...initialScene, observer: { ...initialScene.observer, azimuth: 0.1 } },
        initialScene,
      );
      if (!scene.ok) {
        throw new Error(scene.error);
      }
      return scene.value;
    },
  },
  {
    name: "disk",
    run: () =>
      createDiskProfile(initialScene.space, initialScene.diskInner, initialScene.diskOuter),
  },
  { name: "stars", run: () => createStarTree(createStars()) },
  {
    name: "diffuse",
    run: () => createSkyMap(),
  },
  { name: "thermal", run: () => createBlackbodyTable() },
];

test.for(workloads)("$name preparation", async (workload, { bench, annotate }) => {
  let result: ReturnType<(typeof workloads)[number]["run"]> | undefined;
  await bench(
    `${workload.name} / wall-clock ms`,
    { writeResult: `test-results/bench/cpu-${workload.name}.json` },
    () => {
      result = workload.run();
    },
  ).run({ iterations: 64, time: 1000, warmupIterations: 16, warmupTime: 250 });
  // Retain the actual prepared value, so the measured work produces observable data.
  // Complete serialization and hashing stay outside the measured callback.
  expect(result).toBeDefined();
  const serialized = JSON.stringify(result);
  if (serialized === undefined) {
    throw new Error("Preparation produced no result.");
  }
  const outputSHA256 = createHash("sha256").update(serialized).digest("hex");
  const hash = createHash("sha256");
  for (const directory of ["scene", "physics", "data"]) {
    const url = new URL(`../src/${directory}/`, import.meta.url);
    for (const file of readdirSync(url)
      .filter((name) => name.endsWith(".ts"))
      .toSorted()) {
      hash.update(`${directory}/${file}`);
      hash.update(readFileSync(new URL(file, url)));
    }
  }
  const body = JSON.stringify(
    {
      schema: 1,
      workload: workload.name,
      outputSHA256,
      sourceSHA256: hash.digest("hex"),
      runtime: process.version,
      cpu: cpus()[0]?.model,
      platform: process.platform,
      architecture: process.arch,
      release: release(),
      scene: initialScene,
      sampling: {
        minimumIterations: 64,
        minimumTimeMs: 1000,
        warmupIterations: 16,
        warmupTimeMs: 250,
      },
      scope:
        "Production CPU preparation; no GPU, UI or reference-only trajectory solver. Scene workload reuses unchanged spacetime.",
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
