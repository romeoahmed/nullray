# Nullray

Nullray is an interactive Kerr–Newman observatory built with TypeScript and native WebGPU/WGSL. A dedicated worker traces light through a finite thermal disk and prescribed jets, then forms a photograph from spectral radiation and catalogue stars.

The renderer is an optical prototype. It implements physical observers, signed-radius and horizon continuation, vacuum linear polarization, and a time-dependent narrowband plasma model. General critical-ray accuracy, nonlinear stellar-image completeness and magnetized transfer remain open; [Coverage](coverage.md) separates executable evidence from those limitations.

## Start

Use Node.js 26 or newer, pnpm and a browser meeting the [native platform requirements](tooling.md#native-browser-baseline):

```sh
pnpm install --frozen-lockfile
pnpm dev
```

Choose one of six views, drag the image to explore, and open Settings to change the observer or light sources. Refine freezes the emission epoch and accumulates 64 samples at native resolution. Saved views remain in the browser; a view link carries the physical and display inputs in schema 1.

## Project guide

| Document                             | Owns                                                           |
| ------------------------------------ | -------------------------------------------------------------- |
| [Physics](physics.md)                | Units, equations, source prescriptions and physical limits     |
| [Numerics](numerics.md)              | Coordinates, integration, events, sampling and failure meaning |
| [Architecture](architecture.md)      | Domain boundaries, worker protocol and resource ownership      |
| [Visual design](visual-direction.md) | Composition, interface hierarchy and visual acceptance         |
| [Validation](validation.md)          | Required checks and reference/comparison policy                |
| [Coverage](coverage.md)              | Executable evidence and outstanding numerical coverage         |
| [Tooling](tooling.md)                | Commands, compiler policy and platform requirements            |
| [Agent guide](../AGENTS.md)          | Operational repository instructions                            |
| [Attribution](../NOTICE.md)          | Included data, adapted code and licenses                       |
