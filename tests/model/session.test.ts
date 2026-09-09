import { expect, test } from "vitest";
import * as fc from "fast-check";
import { initialSession, transition } from "../../src/model/session.ts";
import type { Action, Session } from "../../src/model/session.ts";
import { encodeView, decodeView } from "../../src/model/view.ts";

function apply(session: Session, action: Action): Session {
  const result = transition(session, action);
  if (!result.ok) {
    throw new Error(result.error);
  }
  return result.value;
}

test("invalid coupled geometry leaves the complete session untouched", () => {
  const before = encodeView(initialSession.view);
  const result = transition(initialSession, {
    type: "scene",
    value: {
      ...initialSession.view.scene,
      space: { spin: 0.9, charge: 0.9 },
    },
  });
  expect(result.ok).toBe(false);
  expect(encodeView(initialSession.view)).toBe(before);
});

test("presentation edits preserve physical inputs and survive a view round trip", () => {
  fc.assert(
    fc.property(
      fc.double({ min: -6, max: 6, noNaN: true }),
      fc.double({ min: 0, max: 1, noNaN: true }),
      (exposure, bloom) => {
        const changed = apply(apply(initialSession, { type: "exposure", value: exposure }), {
          type: "bloom",
          value: bloom,
        });
        expect(changed.view.scene).toEqual(initialSession.view.scene);
        expect(changed.view.appearance).toEqual(initialSession.view.appearance);
        const restored = decodeView(encodeView(changed.view));
        expect(restored.kind).toBe("view");
        if (restored.kind === "view") {
          expect(restored.value.scene).toEqual(changed.view.scene);
          expect(restored.value.appearance).toEqual(changed.view.appearance);
          // JSON normalizes -0; its exposure and bloom semantics are identical to +0.
          expect(restored.value.exposureEV === exposure).toBe(true);
          expect(restored.value.bloom === bloom).toBe(true);
        }
      },
    ),
    {
      examples: [
        [-0, 0],
        [0, -0],
      ],
    },
  );
});

test("photography freezes motion, diagnostics stop refinement, and restored views stay paused", () => {
  const photo = apply(initialSession, { type: "refine" });
  expect(photo.motion).toBe("refining");
  const diagnostic = apply(photo, { type: "diagnostic", value: "frequency" });
  expect(diagnostic.motion).toBe("paused");
  expect(transition(diagnostic, { type: "refine" }).ok).toBe(false);
  const restored = apply(diagnostic, {
    type: "restore",
    value: { ...initialSession.view, time: 123 },
  });
  expect(restored.motion).toBe("paused");
  expect(restored.view.time).toBe(123);
  expect(apply(restored, { type: "toggle-motion" }).motion).toBe("playing");
});
