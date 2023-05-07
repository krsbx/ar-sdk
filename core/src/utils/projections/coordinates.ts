import { applyModelViewProjectionTransform } from './transforms';

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
