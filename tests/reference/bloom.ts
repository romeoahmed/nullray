type Image = { width: number; height: number; data: Float16Array };
const sample = (image: Image, u: number, v: number, channel: number) => {
  const x = u * image.width - 0.5;
  const y = v * image.height - 0.5;
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  let value = 0;
  for (let dy = 0; dy <= 1; dy++) {
    for (let dx = 0; dx <= 1; dx++) {
      const px = Math.max(0, Math.min(image.width - 1, ix + dx));
      const py = Math.max(0, Math.min(image.height - 1, iy + dy));
      value +=
        (image.data[(py * image.width + px) * 4 + channel] ?? NaN) *
        (dx ? x - ix : 1 - x + ix) *
        (dy ? y - iy : 1 - y + iy);
    }
  }
  return value;
};

/** Direct binary64 tent convolution and linear interpolation, rounded at each HDR target. */
export function referenceBloom(width: number, height: number, data: Float16Array) {
  const filter = (source: Image, w: number, h: number, detail?: Image, detailWeight = 0): Image => {
    const output = new Float16Array(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const u = (x + 0.5) / w;
        const v = (y + 0.5) / h;
        for (let channel = 0; channel < 3; channel++) {
          let value = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              value +=
                (((2 - Math.abs(dx)) * (2 - Math.abs(dy))) / 16) *
                sample(source, u + dx / source.width, v + dy / source.height, channel);
            }
          }
          output[(y * w + x) * 4 + channel] = detail
            ? (1 - detailWeight) * value + detailWeight * sample(detail, u, v, channel)
            : value;
        }
        output[(y * w + x) * 4 + 3] = 1;
      }
    }
    return { width: w, height: h, data: output };
  };
  const levels: Image[] = [];
  const span = Math.max(1, Math.log2(height * 0.08));
  let result = { width, height, data };
  for (let level = 0; level < Math.ceil(span); level++) {
    result = filter(
      result,
      Math.max(1, Math.ceil(result.width / 2)),
      Math.max(1, Math.ceil(result.height / 2)),
    );
    levels.push(result);
    if (result.width === 1 && result.height === 1) {
      break;
    }
  }
  for (let index = levels.length - 2; index >= 0; index--) {
    const detail = levels[index];
    if (detail) {
      result = filter(result, detail.width, detail.height, detail, Math.min(1, 1 / (span - index)));
    }
  }
  return result;
}
