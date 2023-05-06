export type MaximaMinimaPoint = {
  maxima: boolean;
  x: number;
  y: number;
  scale: number;
  angle: number;
  descriptors: number[];
};

export interface Matches {
  querypoint: MaximaMinimaPoint;
  keypoint: MaximaMinimaPoint;
}

export type KeyFrame = {
  maximaPoints: MaximaMinimaPoint[];
  minimaPoints: MaximaMinimaPoint[];
  maximaPointsCluster: {
    rootNode: INode;
  };
  minimaPointsCluster: {
    rootNode: INode;
  };
  width: number;
  height: number;
  scale: number;
};

export type Kernel = {
  variableNames: string[];
  outputShape: number[];
  userCode: string;
};

export interface Crop {
  startX: number;
  startY: number;
  cropSize: number;
}

export type DebugExtra = {
  pyramidImages: number[][];
  dogPyramidImages: number[][] | null[][];
  extremasResults: number[];
  extremaAngles: number[];
  prunedExtremas: number[][];
  localizedExtremas: number[][];
  matches: Matches[];
  matches2: Matches[];
  houghMatches: Matches[];
  houghMatches2: Matches[];
  inlierMatches: Matches[];
  inlierMatches2: Matches[];
  projectedImage: number[][];
  matchingPoints: number[][];
  crop: Crop;
  goodTrack: number[];
  trackedPoints: { x: number; y: number }[];
};

export as namespace ArDetector;
