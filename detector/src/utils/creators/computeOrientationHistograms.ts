import * as tf from '@tensorflow/tfjs';
import { Constants } from '@krsbx/ar-sdk-core';

const {
  ORIENTATION_GAUSSIAN_EXPANSION_FACTOR,
  ORIENTATION_REGION_EXPANSION_FACTOR,
} = Constants.IMAGE_TARGET.DETECTOR;

export function createOrientationHistogramsCache() {
  const gwScale = -1.0 / (2 * ORIENTATION_GAUSSIAN_EXPANSION_FACTOR ** 2);
  const radius =
    ORIENTATION_GAUSSIAN_EXPANSION_FACTOR * ORIENTATION_REGION_EXPANSION_FACTOR;
  const radiusCeil = Math.ceil(radius);

  const radialProperties: number[][] = [];

  for (let y = -radiusCeil; y <= radiusCeil; y++) {
    for (let x = -radiusCeil; x <= radiusCeil; x++) {
      const distance = Math.sqrt(x ** 2 + y ** 2);

      if (distance > radius) continue;

      const w = Math.exp(gwScale * distance ** 2);

      radialProperties.push([x, y, w]);
    }
  }

  return tf.keep(tf.tensor(radialProperties, [radialProperties.length, 3]));
}
