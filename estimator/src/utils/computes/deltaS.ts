import { Matrix, inverse } from 'ml-matrix';

export function getDeltaS(arg: { dU: number[][]; J_U_S: number[][] }) {
  const { dU, J_U_S } = arg;

  const J = new Matrix(J_U_S);
  const U = new Matrix(dU);

  const JT = J.transpose();
  const JTJ = JT.mmul(J);
  const JTU = JT.mmul(U);

  try {
    const JTJInv = inverse(JTJ);
    const S = JTJInv.mmul(JTU);

    return S.to1DArray();
  } catch {
    return null;
  }
}
