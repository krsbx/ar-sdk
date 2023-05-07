/* eslint-disable prefer-destructuring */
/* eslint-disable camelcase */
import { applyModelViewProjectionTransform } from '../helper';

export function getJ_U_S(arg: {
  modelViewProjectionTransform: number[][];
  modelViewTransform: number[][];
  projectionTransform: number[][];
  worldCoord: ArEstimator.Vector3;
  J_U_Xc: number[][];
  J_Xc_S: number[][];
}) {
  const {
    modelViewProjectionTransform,
    modelViewTransform,
    projectionTransform,
    worldCoord,
    J_U_Xc,
    J_Xc_S,
  } = arg;

  const T = modelViewTransform;
  const { x, y } = worldCoord;

  const u = applyModelViewProjectionTransform(
    modelViewProjectionTransform,
    x,
    y
  );

  const z2 = u.z * u.z;
  // Question: This is the most confusing matrix to me. I've no idea how to derive this.
  // J_U_Xc[0][0] = (projectionTransform[0][0] * u.z - projectionTransform[2][0] * u.x) / z2;
  // J_U_Xc[0][1] = (projectionTransform[0][1] * u.z - projectionTransform[2][1] * u.x) / z2;
  // J_U_Xc[0][2] = (projectionTransform[0][2] * u.z - projectionTransform[2][2] * u.x) / z2;
  // J_U_Xc[1][0] = (projectionTransform[1][0] * u.z - projectionTransform[2][0] * u.y) / z2;
  // J_U_Xc[1][1] = (projectionTransform[1][1] * u.z - projectionTransform[2][1] * u.y) / z2;
  // J_U_Xc[1][2] = (projectionTransform[1][2] * u.z - projectionTransform[2][2] * u.y) / z2;

  // The above is the original implementation, but simplify to below becuase projetionTransform[2][0] and [2][1] are zero
  J_U_Xc[0][0] = (projectionTransform[0][0] * u.z) / z2;
  J_U_Xc[0][1] = (projectionTransform[0][1] * u.z) / z2;
  J_U_Xc[0][2] =
    (projectionTransform[0][2] * u.z - projectionTransform[2][2] * u.x) / z2;
  J_U_Xc[1][0] = (projectionTransform[1][0] * u.z) / z2;
  J_U_Xc[1][1] = (projectionTransform[1][1] * u.z) / z2;
  J_U_Xc[1][2] =
    (projectionTransform[1][2] * u.z - projectionTransform[2][2] * u.y) / z2;

  /*
    J_Xc_S should be like this, but z is zero, so we can simplify
    [T[0][2] * y - T[0][1] * z, T[0][0] * z - T[0][2] * x, T[0][1] * x - T[0][0] * y, T[0][0], T[0][1], T[0][2]],
    [T[1][2] * y - T[1][1] * z, T[1][0] * z - T[1][2] * x, T[1][1] * x - T[1][0] * y, T[1][0], T[1][1], T[1][2]],
    [T[2][2] * y - T[2][1] * z, T[2][0] * z - T[2][2] * x, T[2][1] * x - T[2][0] * y, T[2][0], T[2][1], T[2][2]],
  */
  J_Xc_S[0][0] = T[0][2] * y;
  J_Xc_S[0][1] = -T[0][2] * x;
  J_Xc_S[0][2] = T[0][1] * x - T[0][0] * y;
  J_Xc_S[0][3] = T[0][0];
  J_Xc_S[0][4] = T[0][1];
  J_Xc_S[0][5] = T[0][2];

  J_Xc_S[1][0] = T[1][2] * y;
  J_Xc_S[1][1] = -T[1][2] * x;
  J_Xc_S[1][2] = T[1][1] * x - T[1][0] * y;
  J_Xc_S[1][3] = T[1][0];
  J_Xc_S[1][4] = T[1][1];
  J_Xc_S[1][5] = T[1][2];

  J_Xc_S[2][0] = T[2][2] * y;
  J_Xc_S[2][1] = -T[2][2] * x;
  J_Xc_S[2][2] = T[2][1] * x - T[2][0] * y;
  J_Xc_S[2][3] = T[2][0];
  J_Xc_S[2][4] = T[2][1];
  J_Xc_S[2][5] = T[2][2];

  const J_U_S: number[][] = [[], []];

  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 6; i++) {
      J_U_S[j][i] = 0.0;

      for (let k = 0; k < 3; k++) {
        J_U_S[j][i] += J_U_Xc[j][k] * J_Xc_S[k][i];
      }
    }
  }

  return J_U_S;
}

export function getJ_U_Ss(arg: {
  worldCoords: ArEstimator.Vector3[];
  isRobustMode: boolean;
  K2: number;
  E: number[];
  modelViewProjectionTransform: number[][];
  modelViewTransform: number[][];
  projectionTransform: number[][];
  J_U_Xc: number[][];
  J_Xc_S: number[][];
  dxs: number[];
  dys: number[];
}) {
  const {
    modelViewProjectionTransform,
    modelViewTransform,
    projectionTransform,
    isRobustMode,
    worldCoords,
    J_U_Xc,
    J_Xc_S,
    dxs,
    dys,
    K2,
    E,
  } = arg;

  const result = worldCoords.reduce(
    (prev, worldCoord, n) => {
      if (isRobustMode && E[n] > K2) return prev;

      const J_U_S = getJ_U_S({
        modelViewProjectionTransform,
        modelViewTransform,
        projectionTransform,
        worldCoord,
        J_U_Xc,
        J_Xc_S,
      });

      let mult = 1;

      if (isRobustMode) {
        mult = (1.0 - E[n] / K2) * (1.0 - E[n] / K2);

        for (let j = 0; j < 2; j++) {
          for (let i = 0; i < 6; i++) {
            J_U_S[j][i] *= mult;
          }
        }
      }

      prev.dU.push([dxs[n] * mult], [dys[n] * mult]);
      prev.allJ_U_S.push(...J_U_S);

      return prev;
    },
    {
      dU: [] as number[][],
      allJ_U_S: [] as number[][],
    }
  );

  return result;
}
