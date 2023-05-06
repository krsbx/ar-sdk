import _ from 'lodash';
import * as tf from '@tensorflow/tfjs';
import * as kernels from '../kernels';

export function createReductionKernel(extremasResultsT: tf.Tensor[]) {
  // to reduce to amount of data that need to sync back to CPU by 4 times, we apply this trick:
  // the fact that there is not possible to have consecutive maximum/minimum, we can safe combine 4 pixels into 1
  const reductionKernels = extremasResultsT.map((extremasResultT) => {
    const [height, width] = extremasResultT.shape;

    if (_.isNil(height) || _.isNil(width)) return;

    return kernels.applyPrune(height, width);
  });

  return _.compact(reductionKernels);
}

export function createResult(nBuckets: number, nFeatures: number) {
  // combine results into a tensor of:
  //   nBuckets x nFeatures x [score, octave, y, x]
  const curAbsScores: number[][] = [];
  const result: number[][][] = [];

  for (let i = 0; i < nBuckets; i++) {
    result.push([]);
    curAbsScores.push([]);

    for (let j = 0; j < nFeatures; j++) {
      result[i].push([0, 0, 0, 0]);

      curAbsScores[i].push(0);
    }
  }

  return {
    curAbsScores,
    result,
  };
}
