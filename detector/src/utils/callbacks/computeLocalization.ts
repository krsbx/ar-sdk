import _ from 'lodash';

export function computeExtremas(
  prunedExtremasList: number[][],
  pixels: number[][][]
) {
  const localizedExtremas = _.map(prunedExtremasList, (val, id) => {
    const result = _.takeWhile(val, (_, i) => i < 4);

    if (result[0] === 0) return result;

    const pixel = pixels[id];
    const dx = 0.5 * (pixel[1][2] - pixel[1][0]);
    const dy = 0.5 * (pixel[2][1] - pixel[0][1]);
    const dxx = pixel[1][2] + pixel[1][0] - 2 * pixel[1][1];
    const dyy = pixel[2][1] + pixel[0][1] - 2 * pixel[1][1];
    const dxy = 0.25 * (pixel[0][0] + pixel[2][2] - pixel[0][2] - pixel[2][0]);

    const det = dxx * dyy - dxy ** 2;
    const ux = (dyy * -dx + -dxy * -dy) / det;
    const uy = (-dxy * -dx + dxx * -dy) / det;

    const newY = result[2] + uy;
    const newX = result[3] + ux;

    if (Math.abs(det) < 0.0001) return result;

    result[2] = newY;
    result[3] = newX;

    return result;
  });

  return localizedExtremas;
}
