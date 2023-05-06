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
