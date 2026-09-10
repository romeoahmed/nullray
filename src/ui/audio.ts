import { element } from "./elements.ts";

/** Optional local soundtrack. Browser media decoding streams into a lazily opened gain graph. */
export function bindAudio(root: HTMLElement): () => void {
  const subscriptions = new AbortController();
  const { signal } = subscriptions;
  const file = element(root, "#audio-file", HTMLInputElement);
  const play = element(root, "#audio-play", HTMLButtonElement);
  const clear = element(root, "#audio-clear", HTMLButtonElement);
  const volume = element(root, "#audio-volume", HTMLInputElement);
  const output = element(root, "#audio-volume-value", HTMLOutputElement);
  const status = element(root, "#audio-status", HTMLElement);
  // A media element can only be bound to one MediaElementAudioSourceNode in its lifetime.
  const media = new Audio();
  media.preload = "metadata";
  let context: AudioContext | undefined;
  let gain: GainNode | undefined;
  let url: string | undefined;
  let name = "";
  let revision = 0;
  let playing = false;
  let disposed = false;

  function sync() {
    play.disabled = !url;
    clear.disabled = !url;
    play.textContent = playing ? "Pause audio" : "Play audio";
    play.setAttribute("aria-pressed", String(playing));
    output.value = `${Math.round(volume.valueAsNumber * 100)}%`;
    volume.ariaValueText = output.value;
  }

  function pause() {
    revision++;
    playing = false;
    media.pause();
    if (context?.state === "running") {
      void context.suspend();
    }
    sync();
  }

  function release() {
    pause();
    media.removeAttribute("src");
    media.load();
    if (url) {
      URL.revokeObjectURL(url);
    }
    url = undefined;
    name = "";
    file.value = "";
    status.textContent = "Choose a file for local playback.";
    sync();
  }

  async function toggle() {
    if (playing) {
      pause();
      return;
    }
    if (!url || disposed) {
      return;
    }
    const current = ++revision;
    try {
      // Both activation-sensitive calls run directly in the click's user gesture.
      if (!context) {
        context = new AudioContext({ latencyHint: "playback" });
        context.addEventListener(
          "statechange",
          () => {
            // A pending resume can finish after Pause or a file replacement.
            if (!playing && context?.state === "running") {
              void context.suspend();
            }
          },
          { signal },
        );
      }
      if (!gain) {
        gain = new GainNode(context, { gain: volume.valueAsNumber });
        context.createMediaElementSource(media).connect(gain).connect(context.destination);
      }
      const resumed = context.resume();
      const started = media.play();
      playing = true;
      sync();
      await Promise.all([resumed, started]);
      if (disposed || current !== revision) {
        return;
      }
      status.textContent = name;
    } catch {
      if (disposed || current !== revision) {
        return;
      }
      pause();
      status.textContent = "Audio could not play. Try another file or press Play audio again.";
    }
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
      name = selected.name;
      media.src = url;
      status.textContent = name;
      sync();
    },
    { signal },
  );
  play.addEventListener(
    "click",
    () => {
      void toggle();
    },
    { signal },
  );
  clear.addEventListener("click", release, { signal });
  volume.addEventListener(
    "input",
    () => {
      if (context && gain) {
        gain.gain.cancelAndHoldAtTime(context.currentTime);
        gain.gain.setTargetAtTime(volume.valueAsNumber, context.currentTime, 0.02);
      }
      sync();
    },
    { signal },
  );
  media.addEventListener("ended", pause, { signal });
  media.addEventListener(
    "error",
    () => {
      if (!url || disposed || !media.error) {
        return;
      }
      pause();
      status.textContent = "This audio file could not be decoded. Choose another file.";
    },
    { signal },
  );
  sync();
  return () => {
    if (disposed) {
      return;
    }
    disposed = true;
    subscriptions.abort();
    release();
    gain?.disconnect();
    if (context) {
      void context.close();
    }
  };
}
