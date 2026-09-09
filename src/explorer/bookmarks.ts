import { element } from "./elements.ts";
import { encodeView } from "../model/view.ts";
import type { SavedView } from "../model/view.ts";
import { decodeBookmarks } from "../model/bookmarks.ts";
import type { Bookmark } from "../model/bookmarks.ts";
const storageKey = "nullray.views.v1";

/** Own the browser-local view library; every stored snapshot passes the shared view validator. */
export function bindViews(
  root: HTMLElement,
  snapshot: () => SavedView,
  restore: (fragment: string) => void,
  signal: AbortSignal,
): void {
  const name = element(root, "#saved-name", HTMLInputElement);
  const list = element(root, "#saved-views", HTMLSelectElement);
  const save = element(root, "#save-view", HTMLButtonElement);
  const load = element(root, "#load-view", HTMLButtonElement);
  const remove = element(root, "#delete-view", HTMLButtonElement);
  const undo = element(root, "#undo-view", HTMLButtonElement);
  const status = element(root, "#saved-status", HTMLElement);
  let removed: Bookmark | undefined;
  const read = () => decodeBookmarks(localStorage.getItem(storageKey));
  function render(entries: readonly Bookmark[], selected?: string) {
    list.replaceChildren(...entries.map((entry) => new Option(entry.name, entry.name)));
    if (selected !== undefined) {
      list.value = selected;
    }
    list.disabled = load.disabled = remove.disabled = entries.length === 0;
  }
  function commit(entries: readonly Bookmark[], selected?: string) {
    localStorage.setItem(storageKey, JSON.stringify(entries));
    render(entries, selected);
  }
  function act(action: () => void) {
    try {
      action();
    } catch {
      status.textContent =
        "Saved views could not be read or written. Use Share view to keep a link.";
    }
  }
  const options = { signal };
  save.addEventListener(
    "click",
    () =>
      act(() => {
        const title = name.value.trim();
        const current = snapshot();
        if (!title || title.length > 80) {
          status.textContent = "Enter a name of 1–80 characters.";
          return;
        }
        const entries = read();
        if (entries.some((entry) => entry.name === title)) {
          status.textContent = "That name is already saved. Choose another name.";
          return;
        }
        if (entries.length >= 100) {
          status.textContent = "The library is full. Delete a view before saving another.";
          return;
        }
        commit([...entries, { name: title, fragment: encodeView(current) }], title);
        status.textContent = `Saved “${title}” on this browser.`;
        name.value = "";
      }),
    options,
  );
  load.addEventListener(
    "click",
    () =>
      act(() => {
        const entries = read();
        const entry = entries.find((item) => item.name === list.value);
        if (entry) {
          restore(entry.fragment);
          status.textContent = `Opened “${entry.name}”.`;
        } else {
          render(entries);
          status.textContent = "This view is no longer saved.";
        }
      }),
    options,
  );
  remove.addEventListener(
    "click",
    () =>
      act(() => {
        const entries = read();
        const entry = entries.find((item) => item.name === list.value);
        if (!entry) {
          render(entries);
          return;
        }
        commit(entries.filter((item) => item.name !== entry.name));
        removed = entry;
        undo.hidden = false;
        status.textContent = `Deleted “${entry.name}”. You can undo this deletion.`;
      }),
    options,
  );
  undo.addEventListener(
    "click",
    () =>
      act(() => {
        if (!removed) {
          return;
        }
        const entries = read();
        if (entries.length >= 100 || entries.some((entry) => entry.name === removed?.name)) {
          status.textContent = "Cannot restore this name into the current library.";
          return;
        }
        commit([...entries, removed], removed.name);
        status.textContent = `Restored “${removed.name}”.`;
        removed = undefined;
        undo.hidden = true;
      }),
    options,
  );
  window.addEventListener(
    "storage",
    (event) => {
      if (event.key === storageKey || event.key === null) {
        act(() => render(read(), list.value));
      }
    },
    options,
  );
  act(() => render(read()));
}
