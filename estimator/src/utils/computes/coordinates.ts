import { Utils } from '@krsbx/ar-sdk-core';

const { applyModelViewProjectionTransform } = Utils.projections.transforms;

export function computeScreenCoordinate(
  modelViewProjectionTransform: number[][],
  x: number,
  y: number
) {
  const result = applyModelViewProjectionTransform(
    modelViewProjectionTransform,
    x,
    y
  );

  const { x: ux, y: uy, z: uz } = result;

  return { x: ux / uz, y: uy / uz };
}

export function computeScreenCoordinates(arg: {
  worldCoords: ArEstimator.Vector2[];
  screenCoords: ArEstimator.Vector2[];
  modelViewProjectionTransform: number[][];
  E: number[];
  dxs: number[];
  dys: number[];
}) {
  const {
    worldCoords,
    screenCoords,
    modelViewProjectionTransform,
    dxs,
    dys,
    E,
  } = arg;

  for (let n = 0; n < worldCoords.length; n++) {
    const u = computeScreenCoordinate(
      modelViewProjectionTransform,
      worldCoords[n].x,
      worldCoords[n].y
    );

    const dx = screenCoords[n].x - u.x;
    const dy = screenCoords[n].y - u.y;

    dxs[n] = dx;
    dys[n] = dy;

    E[n] = dx ** 2 + dy ** 2;
  }
}
