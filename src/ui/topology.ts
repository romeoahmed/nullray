import { horizons, stationaryLimits } from "../physics/geometry.ts";
import type { Scene } from "../scene/scene.ts";
import type { RayPath } from "../physics/ray-path.ts";

const namespace = "http://www.w3.org/2000/svg";
const y = (theta: number) => 28 + (theta / Math.PI) * 156;

/**
 * Replace SVG contents with a signed-radius/polar-angle coordinate diagram.
 *
 * @remarks
 * Clips distant points to the visible radius range and joins accepted ray samples.
 * It is neither a proper-distance embedding nor a causal diagram; connected
 * sample segments do not locate exact horizon intersections.
 */
export function drawTopology(target: SVGSVGElement, scene: Scene, ray?: RayPath): void {
  const { space, prepared } = scene;
  const structure = horizons(space);
  const radius = prepared.geometry.point.radius;
  const extent = Math.max(2.5, Math.abs(space.spin), Math.abs(space.charge));
  const x = (r: number) => 174 + (r / extent) * 144;
  const nodes: SVGElement[] = [];
  const add = (name: string, attributes: Record<string, string | number>, content?: string) => {
    const node = document.createElementNS(namespace, name);
    for (const [key, value] of Object.entries(attributes)) {
      node.setAttribute(key, String(value));
    }
    if (content !== undefined) {
      node.textContent = content;
    }
    nodes.push(node);
  };
  add("rect", {
    x: x(-extent),
    y: y(0),
    width: x(0) - x(-extent),
    height: 156,
    class: "negative-sheet",
  });
  for (const theta of [0, Math.PI / 2, Math.PI]) {
    add("path", { d: `M${x(-extent)},${y(theta)}H${x(extent)}`, class: "map-grid" });
  }
  add("path", { d: `M${x(0)},${y(0)}V${y(Math.PI)}`, class: "map-grid" });
  for (const branch of [0, 1] as const) {
    let d = "";
    let connected = false;
    for (let sample = 0; sample <= 180; sample++) {
      const theta = (sample / 180) * Math.PI;
      const limits = stationaryLimits(space, theta);
      if (!limits) {
        connected = false;
        continue;
      }
      d += `${connected ? "L" : "M"}${x(limits[branch])},${y(theta)}`;
      connected = true;
    }
    if (d) {
      add("path", { d, class: "stationary-surface" });
    }
  }
  const radii =
    structure.kind === "pair"
      ? [structure.inner, structure.outer]
      : structure.kind === "single"
        ? [structure.outer]
        : structure.kind === "extremal"
          ? [structure.radius]
          : [];
  for (const r of radii) {
    add("path", { d: `M${x(r)},${y(0)}V${y(Math.PI)}`, class: "horizon-surface" });
  }
  if (space.spin === 0) {
    add("path", { d: `M${x(0)},${y(0)}V${y(Math.PI)}`, class: "singular-surface" });
  } else {
    add("circle", { cx: x(0), cy: y(Math.PI / 2), r: 4, class: "singular-surface" });
  }
  const observerX = x(Math.max(-extent, Math.min(extent, radius)));
  if (ray) {
    let d = "";
    for (const point of ray.points) {
      const horizontal = x(Math.max(-extent, Math.min(extent, point.radius)));
      d += `${d ? "L" : "M"}${horizontal},${y(point.inclination)}`;
    }
    add("path", { d, class: "map-ray" });
  }
  add("circle", {
    cx: observerX,
    cy: y(prepared.geometry.point.inclination),
    r: 4,
    class: "map-observer",
  });
  add("text", { x: 30, y: 16 }, "Negative r");
  add("text", { x: 230, y: 16 }, "Positive r");
  add("text", { x: 10, y: 32 }, "N");
  add("text", { x: 10, y: 110 }, "Eq");
  add("text", { x: 10, y: 188 }, "S");
  for (const r of [-extent, 0, extent]) {
    add("text", { x: x(r), y: 204, "text-anchor": "middle" }, `${r.toFixed(1)} M`);
  }
  add(
    "text",
    { x: 174, y: 224, "text-anchor": "middle" },
    `Observer r = ${radius.toFixed(2)} M${Math.abs(radius) > extent ? " · outside map" : ""}`,
  );
  target.replaceChildren(...nodes);
}
