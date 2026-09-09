import "./explorer/style.css";
import { mountApp } from "./explorer/app.ts";
import { element } from "./explorer/elements.ts";

const root = element(document, "#app", HTMLDivElement);
let dispose = mountApp(root);
const isMount = (value: unknown): value is typeof mountApp => typeof value === "function";
if (import.meta.hot) {
  import.meta.hot.accept("./explorer/app.ts", (module) => {
    const mount: unknown = module?.mountApp;
    if (isMount(mount)) {
      const session = dispose();
      dispose = mount(root, session);
    }
  });
  import.meta.hot.dispose(() => dispose());
}
