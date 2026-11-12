import "./ui/style.css";
import { mountApp } from "./ui/app.ts";
import { element } from "./ui/elements.ts";

const root = element(document, "#app", HTMLElement);
let dispose = mountApp(root);
if (import.meta.hot) {
  import.meta.hot.accept("./ui/app.ts", (module) => {
    if (module) {
      const remount: typeof mountApp = module.mountApp;
      dispose = remount(root, dispose());
    }
  });
  import.meta.hot.dispose(() => dispose());
}
