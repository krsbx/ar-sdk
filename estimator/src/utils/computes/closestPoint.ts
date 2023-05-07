/* eslint-disable camelcase */
import { Constants, Utils } from '@krsbx/ar-sdk-core';
import { getDeltaS } from './deltaS';
import { getJ_U_Ss } from './J_U_S';
import { computeErrors } from './errors';
import { updateModelViewTransform } from './transforms';
import { computeScreenCoordinates } from './coordinates';

const { buildModelViewProjectionTransform } = Utils.projections.transforms;

const {
  ICP_BREAK_LOOP_ERROR_RATIO_THRESH,
  ICP_BREAK_LOOP_ERROR_THRESH,
  ICP_MAX_LOOP,
} = Constants.IMAGE_TARGET.ESTIMATION;

function createICPResult(size: number) {
  const E: number[] = new Array(size);
  const dxs: number[] = new Array(size);
  const dys: number[] = new Array(size);

  return {
    dxs,
    dys,
    E,
  };
}

// ICP iteration
// Question: can someone provide theoretical reference / mathematical proof for the following computations?
export function calculateICP(arg: {
  initialModelViewTransform: number[][];
  projectionTransform: number[][];
  worldCoords: ArEstimator.Vector3[];
  screenCoords: ArEstimator.Vector2[];
  inlierProb: number;
  J_U_Xc: number[][];
  J_Xc_S: number[][];
  mat: number[][];
}) {
  const {
    initialModelViewTransform,
    projectionTransform,
    worldCoords,
    screenCoords,
    inlierProb,
    J_U_Xc,
    J_Xc_S,
    mat,
  } = arg;

  const isRobustMode = inlierProb < 1;

  let modelViewTransform = initialModelViewTransform;

  const error = {
    err0: 0.0,
    err1: 0.0,
  };

  const { E, dxs, dys } = createICPResult(worldCoords.length);

  for (let l = 0; l <= ICP_MAX_LOOP; l++) {
    const modelViewProjectionTransform = buildModelViewProjectionTransform(
      projectionTransform,
      modelViewTransform
    );

    computeScreenCoordinates({
      modelViewProjectionTransform,
      screenCoords,
      worldCoords,
      dxs,
      dys,
      E,
    });

    const { K2, err1 } = computeErrors({
      inlierProb,
      isRobustMode,
      worldCoords,
      E,
    });

    if (err1 < ICP_BREAK_LOOP_ERROR_THRESH) break;
    if (l > 0 && err1 / error.err0 > ICP_BREAK_LOOP_ERROR_RATIO_THRESH) break;
    if (l === ICP_MAX_LOOP) break;

    error.err0 = err1;

    const { allJ_U_S, dU } = getJ_U_Ss({
      modelViewProjectionTransform,
      projectionTransform,
      modelViewTransform,
      isRobustMode,
      worldCoords,
      J_U_Xc,
      J_Xc_S,
      dxs,
      dys,
      K2,
      E,
    });

    const dS = getDeltaS({ dU, J_U_S: allJ_U_S });

    if (dS === null) break;

    modelViewTransform = updateModelViewTransform({
      modelViewTransform,
      mat,
      dS,
    });
  }

  return {
    modelViewTransform,
    err: error.err1,
  };
}
