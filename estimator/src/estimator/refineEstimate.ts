/* eslint-disable camelcase */
import _ from 'lodash';
import { Constants } from '@krsbx/ar-sdk-core';
import { computes } from '../utils';

const { TRACKING_THRESH } = Constants.IMAGE_TARGET.ESTIMATION;

export function refineEstimate(arg: {
  initialModelViewTransform: number[][];
  projectionTransform: number[][];
  worldCoords: ArEstimator.Vector3[];
  screenCoords: ArEstimator.Vector2[];
}) {
  const {
    initialModelViewTransform,
    projectionTransform,
    worldCoords,
    screenCoords,
  } = arg;

  const mat: number[][] = [[], [], []];
  const J_U_Xc: number[][] = [[], []]; // 2x3
  const J_Xc_S: number[][] = [[], [], []]; // 3x6

  // Question: shall we normlize the screen coords as well?
  // Question: do we need to normlize the scale as well, i.e. make coords from -1 to 1
  //
  // normalize world coords - reposition them to center of mass
  //   assume z coordinate is always zero (in our case, the image target is planar with z = 0
  const dx =
    worldCoords.reduce((acc, cur) => acc + cur.x, 0) / worldCoords.length;
  const dy =
    worldCoords.reduce((acc, cur) => acc + cur.y, 0) / worldCoords.length;

  const normalizedWorldCoords = _.map(worldCoords, (worldCoord) => ({
    x: worldCoord.x - dx,
    y: worldCoord.y - dy,
    z: worldCoord.z,
  }));

  const diffModelViewTransform: number[][] = [[], [], []];

  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < 3; i++) {
      diffModelViewTransform[j][i] = initialModelViewTransform[j][i];
    }
  }

  diffModelViewTransform[0][3] =
    initialModelViewTransform[0][0] * dx +
    initialModelViewTransform[0][1] * dy +
    initialModelViewTransform[0][3];

  diffModelViewTransform[1][3] =
    initialModelViewTransform[1][0] * dx +
    initialModelViewTransform[1][1] * dy +
    initialModelViewTransform[1][3];

  diffModelViewTransform[2][3] =
    initialModelViewTransform[2][0] * dx +
    initialModelViewTransform[2][1] * dy +
    initialModelViewTransform[2][3];

  // use iterative closest point algorithm to refine the modelViewTransform
  const inlierProbs = [1.0, 0.8, 0.6, 0.4, 0.0];

  let updatedModelViewTransform = diffModelViewTransform; // iteratively update this transform
  let finalModelViewTransform = null;

  for (const inlierProb of inlierProbs) {
    const ret = computes.calculateICP({
      initialModelViewTransform: updatedModelViewTransform,
      worldCoords: normalizedWorldCoords,
      projectionTransform,
      screenCoords,
      inlierProb,
      J_U_Xc,
      J_Xc_S,
      mat,
    });

    updatedModelViewTransform = ret.modelViewTransform;

    if (ret.err < TRACKING_THRESH) {
      finalModelViewTransform = updatedModelViewTransform;
      break;
    }
  }

  if (finalModelViewTransform === null) return null;

  // de-normalize
  finalModelViewTransform[0][3] =
    finalModelViewTransform[0][3] -
    finalModelViewTransform[0][0] * dx -
    finalModelViewTransform[0][1] * dy;
  finalModelViewTransform[1][3] =
    finalModelViewTransform[1][3] -
    finalModelViewTransform[1][0] * dx -
    finalModelViewTransform[1][1] * dy;
  finalModelViewTransform[2][3] =
    finalModelViewTransform[2][3] -
    finalModelViewTransform[2][0] * dx -
    finalModelViewTransform[2][1] * dy;

  return finalModelViewTransform;
}
