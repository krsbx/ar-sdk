/* eslint-disable prefer-destructuring */
/* eslint-disable camelcase */

export function getBaseModelRotation(ra: number, dS: number[]) {
  if (ra < 0.000001)
    return {
      q0: 1.0,
      q1: 0.0,
      q2: 0.0,
      ra: 0.0,
    };

  return {
    ra: Math.sqrt(ra),
    q0: dS[0] / ra,
    q1: dS[1] / ra,
    q2: dS[2] / ra,
  };
}

export function updateModelViewTransform(arg: {
  modelViewTransform: number[][];
  mat: number[][];
  dS: number[];
}) {
  const { modelViewTransform, mat, dS } = arg;

  /**
   * dS has 6 paragrams, first half is rotation, second half is translation
   * rotation is expressed in angle-axis,
   *   [S[0], S[1] ,S[2]] is the axis of rotation, and the magnitude is the angle
   */

  const { q0, q1, q2, ra } = getBaseModelRotation(
    dS[0] ** 2 + dS[1] ** 2 + dS[2] ** 2,
    dS
  );

  const cra = Math.cos(ra);
  const sra = Math.sin(ra);
  const one_cra = 1.0 - cra;

  // mat is [R|t], 3D rotation and translation
  mat[0][0] = q0 * q0 * one_cra + cra;
  mat[0][1] = q0 * q1 * one_cra - q2 * sra;
  mat[0][2] = q0 * q2 * one_cra + q1 * sra;
  mat[0][3] = dS[3];
  mat[1][0] = q1 * q0 * one_cra + q2 * sra;
  mat[1][1] = q1 * q1 * one_cra + cra;
  mat[1][2] = q1 * q2 * one_cra - q0 * sra;
  mat[1][3] = dS[4];
  mat[2][0] = q2 * q0 * one_cra - q1 * sra;
  mat[2][1] = q2 * q1 * one_cra + q0 * sra;
  mat[2][2] = q2 * q2 * one_cra + cra;
  mat[2][3] = dS[5];

  // the updated transform is the original transform x delta transform
  const mat2: number[][] = [[], [], []];

  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < 4; i++) {
      mat2[j][i] =
        modelViewTransform[j][0] * mat[0][i] +
        modelViewTransform[j][1] * mat[1][i] +
        modelViewTransform[j][2] * mat[2][i];
    }

    mat2[j][3] += modelViewTransform[j][3];
  }

  return mat2;
}
