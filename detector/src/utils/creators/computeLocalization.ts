import * as tf from '@tensorflow/tfjs';
import * as runner from '../runner';

export function createResult(
  program: tf.GPGPUProgram,
  prunedExtremasList: number[][],
  dogPyramidImagesT: tf.Tensor[]
) {
  const prunedExtremasT = tf.tensor(
    prunedExtremasList,
    [prunedExtremasList.length, prunedExtremasList[0].length],
    'int32'
  );

  const pixelsT = runner.compileAndRun(program, [
    ...dogPyramidImagesT.slice(1),
    prunedExtremasT,
  ]);

  const pixels = pixelsT.arraySync() as number[][][];

  return pixels;
}
