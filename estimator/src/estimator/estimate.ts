import { Matrix, inverse } from 'ml-matrix';
import { Utils } from '@krsbx/ar-sdk-core';

const { solveHomography } = Utils.homography.solver;

// build world matrix with list of matching worldCoords|screenCoords
//
// Step 1. estimate homography with list of pairs
// Ref: https://www.uio.no/studier/emner/matnat/its/TEK5030/v19/lect/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  (Basic homography estimation from points)
//
// Step 2. decompose homography into rotation and translation matrixes (i.e. world matrix)
// Ref: can anyone provide reference?

export function estimate(arg: {
  screenCoords: ArEstimator.Vector2[];
  worldCoords: ArEstimator.Vector3[];
  projectionTransform: number[][];
}) {
  const { projectionTransform, screenCoords, worldCoords } = arg;

  const Harray = solveHomography(
    worldCoords.map((p) => [p.x, p.y]),
    screenCoords.map((p) => [p.x, p.y])
  );

  if (!Harray) return null;

  const K = new Matrix(projectionTransform);

  const KInv = inverse(K);
  const KInvH = KInv.mmul(Harray).to1DArray();

  const norm1 = Math.sqrt(KInvH[0] ** 2 + KInvH[3] ** 2 + KInvH[6] ** 2);
  const norm2 = Math.sqrt(KInvH[1] ** 2 + KInvH[4] ** 2 + KInvH[7] ** 2);

  const tnorm = (norm1 + norm2) / 2;

  const rotates = [];

  // First Column
  rotates[0] = KInvH[0] / norm1;
  rotates[3] = KInvH[3] / norm1;
  rotates[6] = KInvH[6] / norm1;

  // Second Column
  rotates[1] = KInvH[1] / norm2;
  rotates[4] = KInvH[4] / norm2;
  rotates[7] = KInvH[7] / norm2;

  // Third Column
  rotates[2] = rotates[3] * rotates[7] - rotates[6] * rotates[4];
  rotates[5] = rotates[6] * rotates[1] - rotates[0] * rotates[7];
  rotates[8] = rotates[0] * rotates[4] - rotates[1] * rotates[3];

  const norm3 = Math.sqrt(
    rotates[2] * rotates[2] + rotates[5] * rotates[5] + rotates[8] * rotates[8]
  );
  rotates[2] /= norm3;
  rotates[5] /= norm3;
  rotates[8] /= norm3;

  // TODO: artoolkit has check_rotation() that somehow switch the rotate vector. not sure what that does. Can anyone advice?
  // https://github.com/artoolkitx/artoolkit5/blob/5bf0b671ff16ead527b9b892e6aeb1a2771f97be/lib/SRC/ARICP/icpUtil.c#L215

  const trans = [];
  trans[0] = KInvH[2] / tnorm;
  trans[1] = KInvH[5] / tnorm;
  trans[2] = KInvH[8] / tnorm;

  const initialModelViewTransform = [
    [rotates[0], rotates[1], rotates[2], trans[0]],
    [rotates[3], rotates[4], rotates[5], trans[1]],
    [rotates[6], rotates[7], rotates[8], trans[2]],
  ];

  return initialModelViewTransform;
}
