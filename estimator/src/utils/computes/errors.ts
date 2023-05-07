import { Constants } from '@krsbx/ar-sdk-core';

const { K2_FACTOR } = Constants.IMAGE_TARGET.ESTIMATION;

export function computeErrors(arg: {
  worldCoords: ArEstimator.Vector3[];
  inlierProb: number;
  isRobustMode: boolean;
  E: number[];
}) {
  const { isRobustMode, worldCoords, inlierProb, E } = arg;

  if (!isRobustMode)
    return {
      err1: E.reduce((acc, cur) => acc + cur, 0.0) / worldCoords.length,
      K2: 0,
    };

  let err1 = 0.0;
  let K2 = 0;

  const inlierNum = Math.max(
    3,
    Math.floor(worldCoords.length * inlierProb) - 1
  );

  E.sort((a, b) => a - b);

  K2 = Math.max(E[inlierNum] * K2_FACTOR, 16.0);

  for (let n = 0; n < worldCoords.length; n++) {
    if (E[n] > K2) {
      err1 += K2 / 6;
      continue;
    }

    err1 += (K2 / 6.0) * (1.0 - (1.0 - E[n] / K2) ** 3);
  }

  err1 /= worldCoords.length;

  return {
    err1: err1 / worldCoords.length,
    K2,
  };
}
