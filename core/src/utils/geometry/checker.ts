import { linePointSide } from './helper';

// srcPoints, dstPoints: array of four elements [x, y]
export function checkFourPointsConsistent(arg: {
  x1: number[];
  x2: number[];
  x3: number[];
  x4: number[];
  x1p: number[];
  x2p: number[];
  x3p: number[];
  x4p: number[];
}) {
  const { x1, x1p, x2, x2p, x3, x3p, x4, x4p } = arg;

  if (linePointSide(x1, x2, x3) > 0 !== linePointSide(x1p, x2p, x3p) > 0)
    return false;
  if (linePointSide(x2, x3, x4) > 0 !== linePointSide(x2p, x3p, x4p) > 0)
    return false;
  if (linePointSide(x3, x4, x1) > 0 !== linePointSide(x3p, x4p, x1p) > 0)
    return false;
  if (linePointSide(x4, x1, x2) > 0 !== linePointSide(x4p, x1p, x2p) > 0)
    return false;

  return true;
}

export function checkThreePointsConsistent(arg: {
  x1: number[];
  x2: number[];
  x3: number[];
  x1p: number[];
  x2p: number[];
  x3p: number[];
}) {
  const { x1, x2, x3, x1p, x2p, x3p } = arg;

  if (linePointSide(x1, x2, x3) > 0 !== linePointSide(x1p, x2p, x3p) > 0)
    return false;

  return true;
}

// check if four points form a convex quadrilaternal.
// all four combinations should have same sign
export function quadrilateralConvex(arg: {
  x1: number[];
  x2: number[];
  x3: number[];
  x4: number[];
}) {
  const { x1, x2, x3, x4 } = arg;

  const first = linePointSide(x1, x2, x3) <= 0;

  if (linePointSide(x2, x3, x4) <= 0 !== first) return false;
  if (linePointSide(x3, x4, x1) <= 0 !== first) return false;
  if (linePointSide(x4, x1, x2) <= 0 !== first) return false;

  return true;
}
