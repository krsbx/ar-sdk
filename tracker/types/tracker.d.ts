export type TrackingFeature = ImageData & {
  points: { x: number; y: number }[];
};

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

export as namespace ArTracker;
