import { Matrix, determinant, inverse } from 'ml-matrix';
import { transformToMatrixLikeMxN } from './helper';

export function matrixInverse33(A: number[], threshold: number) {
  const matA = new Matrix(transformToMatrixLikeMxN(A, 3, 3));
  const det = determinant(matA);

  if (Math.abs(det) <= threshold) return null;

  return inverse(matA).to1DArray();
}

export function matrixMul33(A: number[], B: number[]) {
  const matA = new Matrix(transformToMatrixLikeMxN(A, 3, 3));
  const matB = new Matrix(transformToMatrixLikeMxN(B, 3, 3));

  return matA.mmul(matB).to1DArray();
}

export function multiplyPointHomographyInhomogenous(x: number[], H: number[]) {
  const w = H[6] * x[0] + H[7] * x[1] + H[8];

  const xp = [
    (H[0] * x[0] + H[1] * x[1] + H[2]) / w,
    (H[3] * x[0] + H[4] * x[1] + H[5]) / w,
  ];

  return xp;
}
