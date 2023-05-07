import * as tf from '@tensorflow/tfjs';

export function compileAndRun(
  program: tf.GPGPUProgram,
  inputs: tf.TensorInfo[]
) {
  const outInfo = (tf.backend() as tf.MathBackendWebGL).compileAndRun(
    program,
    inputs
  );

  return tf.engine().makeTensorFromTensorInfo(outInfo);
}

export function runWebGlProgram(
  program: tf.GPGPUProgram,
  inputs: tf.TensorInfo[],
  outputType: keyof tf.DataTypeMap
) {
  // Reuse the backend and engine
  // By doing this we doesnt need to create a new backend and engine for each detection
  const outInfo = (tf.backend() as tf.MathBackendWebGL).runWebGLProgram(
    program,
    inputs,
    outputType
  );

  return tf.engine().makeTensorFromTensorInfo(outInfo);
}

export function preBuild(
  trackingFrame: ArTracker.TrackingFeature,
  maxCount: number
) {
  return tf.tidy(() => {
    const { scale } = trackingFrame;

    const p: number[][] = [];

    for (let k = 0; k < maxCount; k++) {
      if (k < trackingFrame.points.length) {
        p.push([
          trackingFrame.points[k].x / scale,
          trackingFrame.points[k].y / scale,
        ]);
      } else {
        p.push([-1, -1]);
      }
    }

    const imagePixels = tf.tensor(trackingFrame.data, [
      trackingFrame.width * trackingFrame.height,
    ]);

    const imageProperties = tf.tensor(
      [trackingFrame.width, trackingFrame.height, trackingFrame.scale],
      [3]
    );

    const featurePoints = tf.tensor(p, [p.length, 2], 'float32');

    return {
      featurePoints,
      imagePixels,
      imageProperties,
    };
  });
}
