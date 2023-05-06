export function transformToMatrixLikeMxN(A: number[], m: number, n: number) {
  const matA: number[][] = [];

  for (let i = 0; i < m; i++) {
    const id = i * n;
    const mat: number[] = [];

    for (let j = 0; j < n; j++) {
      mat.push(A[id + j]);
    }

    matA.push(mat);
  }

  return matA;
}

export function vector2DDistance(vecA: number[], vecB: number[]) {
  return [vecA[0] - vecB[0], vecA[1] - vecB[1]];
}

export function areaOfTriangle(u: number[], v: number[]) {
  const a = u[0] * v[1] - u[1] * v[0];

  return Math.abs(a) * 0.5;
}

// check which side point C on the line from A to B
export function linePointSide(A: number[], B: number[], C: number[]) {
  return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]);
}
