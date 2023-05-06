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
