import type { Action, Session } from "../model/session.ts";
import { outerHorizon } from "../physics/spacetime.ts";
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
    bounds?: (state: Session) => readonly [number, number],
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
      const [low, high] = bounds?.(state) ?? [minimum, maximum];
      input.min = String(Math.min(low, value));
      input.max = String(Math.max(high, value));
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
        value: { ...view.scene, space: { ...view.scene.space, [key]: value } },
      }),
      (value) => value.toFixed(2),
      ({ view }) => {
        const other = view.scene.space[key === "spin" ? "charge" : "spin"];
        const maximum = Math.floor(100 * Math.sqrt(Math.max(0, 0.99 ** 2 - other ** 2))) / 100;
        return [-maximum, maximum];
      },
    );
  }
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
    ({ view }) => [Math.ceil(10 * (outerHorizon(view.scene.space) + 0.05)) / 10, 200],
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
    "navigation",
    ["orbit", "free"],
    ({ view }) => view.navigation,
    (value) => ({ type: "navigation", value }),
  );
  select(
    "display-mode",
    ["auto", "hdr", "sdr"],
    ({ view }) => view.display,
    (value) => ({ type: "display", value }),
  );
  select(
    "diagnostic",
    ["image", "frequency", "order"],
    ({ view }) => view.diagnostic,
    (value) => ({ type: "diagnostic", value }),
  );
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
