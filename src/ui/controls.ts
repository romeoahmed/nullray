import { initialJet } from "../scene/jet.ts";
import { initialPlasma } from "../scene/plasma.ts";
import type { Action, Session } from "../scene/session.ts";
import type { PhysicalObserver } from "../scene/preparation.ts";
import { element } from "./elements.ts";

const degrees = (radians: number) => (radians * 180) / Math.PI;
const radians = (angle: number) => (angle * Math.PI) / 180;

/**
 * Bind native controls to domain actions; the DOM never becomes the source of scene state.
 * @returns A synchronizer to update controls after either accepting or rejecting an action.
 */
export function bindControls(
  root: HTMLElement,
  current: () => Session,
  dispatch: (action: Action) => void,
  signal: AbortSignal,
): () => void {
  const sync: (() => void)[] = [];
  const range = (
    id: string,
    read: (state: Session) => number,
    action: (state: Session, value: number) => Action,
    format: (value: number) => string,
  ) => {
    const input = element(root, `#${id}`, HTMLInputElement);
    const output = element(root, `#${id}-value`, HTMLOutputElement);
    input.addEventListener("input", () => dispatch(action(current(), input.valueAsNumber)), {
      signal,
    });
    const minimum = Number(input.min),
      maximum = Number(input.max);
    sync.push(() => {
      const state = current(),
        value = read(state);
      input.min = String(Math.min(minimum, value));
      input.max = String(Math.max(maximum, value));
      input.value = String(value);
      const description = format(value);
      output.value = description;
      input.ariaValueText = description;
    });
  };
  const select = <T extends string>(
    id: string,
    choices: readonly T[],
    read: (state: Session) => T,
    action: (value: T) => Action,
  ) => {
    const input = element(root, `#${id}`, HTMLSelectElement);
    input.addEventListener(
      "change",
      () => {
        const value = choices.find((choice) => choice === input.value);
        if (value !== undefined) {
          dispatch(action(value));
        }
      },
      { signal },
    );
    sync.push(() => {
      input.value = read(current());
    });
  };
  for (const key of ["spin", "charge"] as const) {
    range(
      key,
      ({ view }) => view.scene.space[key],
      ({ view }, value) => ({
        type: "scene",
        value: {
          observer: view.scene.observer,
          jet: view.scene.jet,
          plasma: view.scene.plasma,
          camera: view.scene.camera,
          space: { ...view.scene.space, [key]: value },
        },
      }),
      (value) => value.toFixed(2),
    );
  }
  select(
    "navigation",
    ["orbit", "free"],
    ({ view }) => view.navigation,
    (value) => ({ type: "navigation", value }),
  );
  select(
    "observer-model",
    ["zamo", "static", "regular", "freefall", "custom"],
    ({ view }) => view.scene.observer.motion.kind,
    (kind) => {
      const { scene } = current().view;
      const u = scene.prepared.frame.velocity;
      const motion: PhysicalObserver =
        kind === "freefall"
          ? { kind, velocity: [0, 0, 0], properTime: 0 }
          : kind === "custom"
            ? { kind, velocity: [u[1] / u[0], u[2] / u[0], u[3] / u[0]] }
            : { kind };
      return { type: "scene", value: { ...scene, observer: { ...scene.observer, motion } } };
    },
  );
  select(
    "observer-chart",
    ["ingoing", "outgoing"],
    ({ view }) => view.scene.observer.chart,
    (chart) => {
      const { scene } = current().view;
      return { type: "scene", value: { ...scene, observer: { ...scene.observer, chart } } };
    },
  );
  const velocityGroup = element(root, "#observer-velocity", HTMLDivElement);
  const clockGroup = element(root, "#observer-clock", HTMLDivElement);
  const velocityHint = element(root, "#velocity-hint", HTMLParagraphElement);
  for (const axis of [0, 1, 2] as const) {
    const input = element(root, `#velocity-${axis}`, HTMLInputElement);
    const label = element(root, `#velocity-${axis}-label`, HTMLLabelElement);
    input.addEventListener(
      "change",
      () => {
        const { scene } = current().view;
        const motion = scene.observer.motion;
        if (motion.kind !== "custom" && motion.kind !== "freefall") {
          return;
        }
        const velocity: [number, number, number] = [...motion.velocity];
        velocity[axis] = input.valueAsNumber;
        dispatch({
          type: "scene",
          value: {
            ...scene,
            observer: {
              ...scene.observer,
              motion: { ...motion, velocity },
            },
          },
        });
      },
      { signal },
    );
    sync.push(() => {
      const motion = current().view.scene.observer.motion;
      input.value = String(
        motion.kind === "custom" || motion.kind === "freefall" ? motion.velocity[axis] : 0,
      );
      label.textContent =
        motion.kind === "freefall"
          ? (
              ["Local radial velocity", "Local polar velocity", "Local azimuthal velocity"] as const
            )[axis]
          : (["Velocity X", "Velocity Y", "Velocity Z"] as const)[axis];
    });
  }
  sync.push(() => {
    const motion = current().view.scene.observer.motion;
    velocityGroup.hidden = motion.kind !== "custom" && motion.kind !== "freefall";
    clockGroup.hidden = motion.kind !== "freefall";
    velocityHint.textContent =
      motion.kind === "freefall"
        ? "Launch velocity in the local regular frame, in units c. Proper time advances a neutral geodesic; the camera stays aligned with its reconstructed rest frame."
        : "d(X,Y,Z)/dT in the selected Cartesian Kerr–Schild chart. The complete velocity must be future timelike.";
  });
  range(
    "proper-time",
    ({ view }) =>
      view.scene.observer.motion.kind === "freefall" ? view.scene.observer.motion.properTime : 0,
    ({ view }, properTime) => {
      const motion = view.scene.observer.motion;
      return {
        type: "scene",
        value: {
          ...view.scene,
          observer: {
            ...view.scene.observer,
            motion: motion.kind === "freefall" ? { ...motion, properTime } : motion,
          },
        },
      };
    },
    (value) => `${value.toFixed(2)} M`,
  );
  range(
    "inclination",
    ({ view }) => degrees(view.scene.observer.inclination),
    ({ view }, value) => ({
      type: "scene",
      value: { ...view.scene, observer: { ...view.scene.observer, inclination: radians(value) } },
    }),
    (value) => `${value.toFixed(1)}°`,
  );
  range(
    "distance",
    ({ view }) => view.scene.observer.radius,
    ({ view }, value) => ({
      type: "scene",
      value: { ...view.scene, observer: { ...view.scene.observer, radius: value } },
    }),
    (value) => `${value.toFixed(1)} rɢ`,
  );
  range(
    "fov",
    ({ view }) => degrees(view.scene.observer.fieldOfView),
    ({ view }, value) => ({
      type: "scene",
      value: { ...view.scene, observer: { ...view.scene.observer, fieldOfView: radians(value) } },
    }),
    (value) => `${value.toFixed(0)}°`,
  );
  range(
    "white-balance",
    ({ view }) => view.whiteBalance,
    (_, value) => ({ type: "white-balance", value }),
    (value) => `${value.toLocaleString("en")} K`,
  );
  for (const { id, key, format } of [
    {
      id: "disk-temperature",
      key: "diskTemperature",
      format: (value: number) => `${Math.round(value).toLocaleString("en")} K`,
    },
    {
      id: "disk-structure",
      key: "diskStructure",
      format: (value: number) => `${Math.round(value * 100)}%`,
    },
    {
      id: "disk-thickness",
      key: "diskThickness",
      format: (value: number) =>
        value === 0 ? "Equatorial surface" : `${(value * 100).toFixed(1)}% peak H/r`,
    },
    {
      id: "disk-optical-depth",
      key: "diskOpticalDepth",
      format: (value: number) => value.toFixed(1),
    },
    {
      id: "other-universes",
      key: "otherUniverses",
      format: (value: number) => (value === 0 ? "Unilluminated" : `${Math.round(100 * value)}%`),
    },
    {
      id: "sky-brightness",
      key: "skyBrightness",
      format: (value: number) => `${value.toFixed(2)}×`,
    },
  ] as const) {
    range(
      id,
      ({ view }) => view.appearance[key],
      ({ view }, value) => ({ type: "appearance", value: { ...view.appearance, [key]: value } }),
      format,
    );
  }
  select(
    "jet-mode",
    ["off", "on"],
    ({ view }) => (view.scene.jet ? "on" : "off"),
    (value) => ({
      type: "scene",
      value: { ...current().view.scene, jet: value === "on" ? initialJet : null },
    }),
  );
  const jetControls = element(root, "#jet-controls", HTMLDivElement);
  sync.push(() => {
    jetControls.hidden = current().view.scene.jet === null;
  });
  select(
    "plasma-mode",
    ["off", "on"],
    ({ view }) => (view.scene.plasma ? "on" : "off"),
    (value) => ({
      type: "scene",
      value: { ...current().view.scene, plasma: value === "on" ? initialPlasma : null },
    }),
  );
  const plasmaControls = element(root, "#plasma-controls", HTMLDivElement);
  sync.push(() => {
    plasmaControls.hidden = current().view.scene.plasma === null;
  });
  for (const key of ["density", "frequencyGHz", "radius"] as const) {
    range(
      `plasma-${key}`,
      ({ view }) => Math.log10((view.scene.plasma ?? initialPlasma)[key]),
      ({ view }, exponent) => ({
        type: "scene",
        value: {
          ...view.scene,
          plasma: { ...(view.scene.plasma ?? initialPlasma), [key]: 10 ** exponent },
        },
      }),
      (value) =>
        key === "density"
          ? `10^${value.toFixed(1)} cm⁻³`
          : `${(10 ** value).toFixed(1)} ${key === "radius" ? "M" : "GHz"}`,
    );
  }
  range(
    "plasma-heating",
    ({ view }) => view.scene.plasma?.heating ?? 0,
    ({ view }, heating) => ({
      type: "scene",
      value: { ...view.scene, plasma: { ...(view.scene.plasma ?? initialPlasma), heating } },
    }),
    (value) =>
      value === 0 ? "Uniform temperature" : `Up to ${(1 + value).toFixed(1)} × 100,000 K`,
  );
  range(
    "jet-energy",
    ({ view }) => Math.log10(view.scene.jet?.gammaMin ?? initialJet.gammaMin),
    ({ view }, exponent) => ({
      type: "scene",
      value: {
        ...view.scene,
        jet: { ...(view.scene.jet ?? initialJet), gammaMin: 10 ** exponent },
      },
    }),
    (value) => `γ ≥ ${Math.round(10 ** value).toLocaleString("en")}`,
  );
  range(
    "jet-speed",
    ({ view }) => view.scene.jet?.speed ?? initialJet.speed,
    ({ view }, speed) => ({
      type: "scene",
      value: { ...view.scene, jet: { ...(view.scene.jet ?? initialJet), speed } },
    }),
    (value) => `${value.toFixed(2)} c`,
  );
  range(
    "jet-opening",
    ({ view }) => degrees(view.scene.jet?.openingAngle ?? initialJet.openingAngle),
    ({ view }, angle) => ({
      type: "scene",
      value: {
        ...view.scene,
        jet: { ...(view.scene.jet ?? initialJet), openingAngle: radians(angle) },
      },
    }),
    (value) => `${value.toFixed(0)}°`,
  );
  range(
    "jet-density",
    ({ view }) => Math.log10(view.scene.jet?.density ?? initialJet.density),
    ({ view }, exponent) => ({
      type: "scene",
      value: { ...view.scene, jet: { ...(view.scene.jet ?? initialJet), density: 10 ** exponent } },
    }),
    (value) => `10^${value.toFixed(1)} cm⁻³`,
  );
  range(
    "exposure",
    ({ view }) => view.exposureEV,
    (_, value) => ({ type: "exposure", value }),
    (value) => `${value > 0 ? "+" : ""}${value.toFixed(1)} EV`,
  );
  range(
    "bloom",
    ({ view }) => view.bloom,
    (_, value) => ({ type: "bloom", value }),
    (value) => `${Math.round(value * 100)}%`,
  );
  select(
    "display-mode",
    ["auto", "hdr", "sdr"],
    ({ view }) => view.display,
    (value) => ({ type: "display", value }),
  );
  select(
    "diagnostic",
    ["image", "frequency", "order", "domain", "polarization", "angle"],
    ({ view }) => view.diagnostic,
    (value) => ({ type: "diagnostic", value }),
  );
  select(
    "analyzer-mode",
    ["off", "linear"],
    ({ view }) => (view.analyzer === null ? "off" : "linear"),
    (value) => ({ type: "analyzer", value: value === "off" ? null : 0 }),
  );
  range(
    "analyzer-angle",
    ({ view }) => degrees(view.analyzer ?? 0),
    (_, value) => ({ type: "analyzer", value: radians(value) }),
    (value) => `${value.toFixed(0)}°`,
  );
  const angleControl = element(root, "#analyzer-angle-control", HTMLDivElement);
  sync.push(() => {
    angleControl.hidden = current().view.analyzer === null;
  });
  select(
    "exploration-resolution",
    ["1", "0.75", "0.5"],
    ({ resolution }) => String(resolution),
    (value) => ({ type: "resolution", value: Number(value) }),
  );
  return () => {
    for (const update of sync) {
      update();
    }
  };
}
