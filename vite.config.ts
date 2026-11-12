import { defineConfig } from "vite";

// The application ships native syntax (explicit resource management included) without lowering.
export default defineConfig({ build: { target: "esnext" } });
