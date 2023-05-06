import _ from 'lodash';
import { Constants } from '@krsbx/ar-sdk-core';

const DETECTOR_CONSTANTS = Constants.IMAGE_TARGET.DETECTOR;

export function computeResult(arg: {
  curAbsScores: number[][];
  result: number[][][];
  reduced: number[][];
  octave: number;
  nFeatures: number;
  width: number;
  height: number;
}) {
  const { curAbsScores, nFeatures, octave, result, height, width, reduced } =
    arg;

  const bucketWidth =
    (width * 2) / DETECTOR_CONSTANTS.NUM_BUCKETS_PER_DIMENSION;
  const bucketHeight =
    (height * 2) / DETECTOR_CONSTANTS.NUM_BUCKETS_PER_DIMENSION;

  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      const encoded = reduced[i][j];

      if (encoded === 0) continue;

      const score = encoded % 1000;
      const loc = Math.floor(Math.abs(encoded) / 1000);
      const x = j * 2 + (loc === 2 || loc === 3 ? 1 : 0);
      const y = i * 2 + (loc === 1 || loc === 3 ? 1 : 0);

      const bucketX = Math.floor(x / bucketWidth);
      const bucketY = Math.floor(y / bucketHeight);

      const bucket =
        bucketY * DETECTOR_CONSTANTS.NUM_BUCKETS_PER_DIMENSION + bucketX;
      const absScore = Math.abs(score);

      let tIndex = nFeatures;

      while (tIndex >= 1 && absScore >= curAbsScores[bucket][tIndex - 1])
        tIndex--;

      if (tIndex >= nFeatures) continue;

      for (let t = nFeatures - 1; t >= tIndex + 1; t--) {
        curAbsScores[bucket][t] = curAbsScores[bucket][t - 1];
        result[bucket][t] = _.cloneDeep(result[bucket][t - 1]);
      }

      curAbsScores[bucket][tIndex] = absScore;
      result[bucket][tIndex][0] = score;
      result[bucket][tIndex][1] = octave;
      result[bucket][tIndex][2] = y;
      result[bucket][tIndex][3] = x;
    }
  }
}
