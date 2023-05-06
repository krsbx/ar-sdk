import _ from 'lodash';
import * as tf from '@tensorflow/tfjs';

export function createFreakDescriptors(extremaFreaks: tf.Tensor) {
  const in1Arr: number[] = [];
  const in2Arr: number[] = [];

  const [, width] = extremaFreaks.shape;

  if (_.isNil(width)) return;

  for (let k1 = 0; k1 < width; k1++) {
    for (let k2 = k1 + 1; k2 < width; k2++) {
      in1Arr.push(k1);
      in2Arr.push(k2);
    }
  }

  const in1 = tf.tensor(in1Arr, [in1Arr.length]).cast('int32');
  const in2 = tf.tensor(in2Arr, [in2Arr.length]).cast('int32');

  return tf.keep(tf.stack([in1, in2], 1));
}
