import { Matrix } from 'ml-matrix';
import { transformToMatrixLikeMxN } from '../geometry/helper';

// centroid at origin and avg distance from origin is sqrt(2)
// skip normalization
// return {normalizedCoords: coords, param: {meanX: 0, meanY: 0, s: 1}};

export function normalizePoints(coords: number[][]) {
  const { meanX, meanY } = coords.reduce(
    (prev, curr, index) => {
      const [x, y] = curr;

      prev.sumX += x;
      prev.sumY += y;

      if (index !== coords.length - 1) return prev;

      prev.meanX = prev.sumX / coords.length;
      prev.meanY = prev.sumY / coords.length;

      return prev;
    },
    {
      sumX: 0,
      sumY: 0,
      meanX: 0,
      meanY: 0,
      sumDiff: 0,
    }
  );

  const sumDiff = coords.reduce((prev, curr) => {
    const [x, y] = curr;

    const diffX = x - meanX;
    const diffY = y - meanY;

    return prev + Math.sqrt(diffX ** 2 + diffY ** 2);
  }, 0);

  const s = (Math.sqrt(2) * coords.length) / sumDiff;

  const normPoints = coords.reduce((prev, curr) => {
    const [x, y] = curr;

    prev.push([(x - meanX) * s, (y - meanY) * s]);

    return prev;
  }, [] as number[][]);

  return { normPoints, param: { meanX, meanY, s } };
}

// Denormalize homography
// where T is the normalization matrix, i.e.
//
//     [1  0  -meanX]
// T = [0  1  -meanY]
//     [0  0     1/s]
//
//          [1  0  s*meanX]
// inv(T) = [0  1  s*meanY]
// 	    [0  0        s]
//
// H = inv(Tdst) * Hn * Tsrc
//
// @param {
//   nH: normH,
//   srcParam: param of src transform,
//   dstParam: param of dst transform
// }

export function denormalizeHomography<
  T extends ReturnType<typeof normalizePoints>['param']
>(nH: number[], srcParam: T, dstParam: T) {
  const normH = new Matrix(transformToMatrixLikeMxN(nH, 3, 3));
  // Set nH[8] = 1
  normH.set(2, 2, 1);

  const srcT = new Matrix([
    [1, 0, -srcParam.meanX],
    [0, 1, -srcParam.meanY],
    [0, 0, 1 / srcParam.s],
  ]);

  const invTdst = new Matrix([
    [1, 0, dstParam.s * dstParam.meanX],
    [0, 1, dstParam.s * dstParam.meanY],
    [0, 0, dstParam.s],
  ]);

  const H = invTdst.mmul(normH).mmul(srcT);

  return H;
}
