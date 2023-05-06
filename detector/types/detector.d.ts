export type MaximaMinimaPoint = {
  maxima: boolean;
  x: number;
  y: number;
  scale: number;
  angle: number;
  descriptors: number[];
};

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

export as namespace ArDetector;
