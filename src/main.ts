import "./ui/style.css";
import { mountApp } from "./ui/app.ts";
import { element } from "./ui/elements.ts";

const root = element(document, "#app", HTMLElement);
let dispose = mountApp(root);
const isMount = (value: unknown): value is typeof mountApp => typeof value === "function";
if (import.meta.hot) {
  import.meta.hot.accept("./ui/app.ts", (module) => {
    const mount: unknown = module?.mountApp;
    if (isMount(mount)) {
      const session = dispose();
      dispose = mount(root, session);
    }
  });
  import.meta.hot.dispose(() => dispose());
}
