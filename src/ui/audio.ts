import { element } from "./elements.ts";

/**
 * Bind local audio selection and own its object URL and event subscriptions.
 *
 * @returns Cleanup that stops playback, clears the source, revokes its URL, and
 * removes listeners. Native media controls handle seeking, volume, and activation.
 */
export function bindAudio(root: HTMLElement): () => void {
  const subscriptions = new AbortController();
  const { signal } = subscriptions;
  const file = element(root, "#audio-file", HTMLInputElement);
  const media = element(root, "#audio-player", HTMLAudioElement);
  const clear = element(root, "#audio-clear", HTMLButtonElement);
  const status = element(root, "#audio-status", HTMLElement);
  let url: string | undefined;
  media.volume = 0.5;

  function release() {
    media.pause();
    media.removeAttribute("src");
    media.load();
    if (url) {
      URL.revokeObjectURL(url);
    }
    url = undefined;
    file.value = "";
    clear.disabled = true;
    media.hidden = true;
    status.textContent = "Choose a soundtrack. Your file stays on this device.";
  }
  file.addEventListener(
    "change",
    () => {
      const selected = file.files?.[0];
      if (!selected) {
        return;
      }
      release();
      url = URL.createObjectURL(selected);
      media.src = url;
      media.hidden = false;
      clear.disabled = false;
      status.textContent = selected.name;
    },
    { signal },
  );
  clear.addEventListener("click", release, { signal });
  media.addEventListener(
    "error",
    () => {
      if (url && media.error) {
        status.textContent = "This file could not be played. Choose another audio file.";
      }
    },
    { signal },
  );
  release();
  return () => {
    subscriptions.abort();
    release();
  };
}
