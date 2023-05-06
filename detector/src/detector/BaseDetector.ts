import _ from 'lodash';
import * as tf from '@tensorflow/tfjs';
import { Constants, Kernels } from '@krsbx/ar-sdk-core';
import { callbacks, creators, runner } from '../utils';

const DETECTOR_CONSTANTS = Constants.IMAGE_TARGET.DETECTOR;
const { FREAKPOINTS } = Constants.IMAGE_TARGET.FREAK;

class BaseDetector {
  protected debugMode: boolean;
  protected width: number;
  protected height: number;
  protected numOctaves: number;
  protected nBuckets: number;
  protected nFeatures: number;

  protected tensorCaches: {
    computeExtremaFreak?: {
      freakPointsT: tf.Tensor<tf.Rank>;
    };
    computeFreakDescriptors?: {
      positionT: tf.Tensor<tf.Rank>;
    };
    orientationHistograms?: {
      radialPropertiesT: tf.Tensor<tf.Rank>;
    };
  };
  protected kernelCaches: {
    applyPrune?: {
      reductionKernels: tf.GPGPUProgram[];
    };
    applyFilter?: Record<string, ArDetector.Kernel[]>;
    buildExtremas?: Record<string, ArDetector.Kernel>;
    computeLocalization?: ArDetector.Kernel[];
    computeOrientationHistograms?: ArDetector.Kernel[];
    computeExtremaAngles?: ArDetector.Kernel;
    computeExtremaFreak?: ArDetector.Kernel[];
    computeFreakDescriptors?: ArDetector.Kernel[];
    upsampleBilinear?: Record<string, ArDetector.Kernel>;
    downsampleBilinear?: Record<string, ArDetector.Kernel>;
    smoothHistograms?: ArDetector.Kernel;
  };

  constructor(arg: { width: number; height: number; debugMode?: boolean }) {
    this.width = arg.width;
    this.height = arg.height;
    this.debugMode = arg?.debugMode ?? false;
    this.numOctaves = DETECTOR_CONSTANTS.PYRAMID_MAX_OCTAVE;

    this.nBuckets = DETECTOR_CONSTANTS.NUM_BUCKETS_PER_DIMENSION ** 2;
    this.nFeatures = DETECTOR_CONSTANTS.MAX_FEATURES_PER_BUCKET;

    this.tensorCaches = {};
    this.kernelCaches = {};
  }

  protected applyPrune(extremasResultsT: tf.Tensor[]) {
    // Create reduction kernel if not exits
    if (!this.kernelCaches?.applyPrune) {
      this.kernelCaches.applyPrune = {
        reductionKernels:
          creators.applyPrune.createReductionKernel(extremasResultsT),
      };
    }

    const { curAbsScores, result } = creators.applyPrune.createResult(
      this.nBuckets,
      this.nFeatures
    );

    tf.tidy(() => {
      if (!this.kernelCaches?.applyPrune?.reductionKernels) return;

      const { reductionKernels } = this.kernelCaches.applyPrune;

      callbacks.applyPrune.computePrune({
        nFeatures: this.nFeatures,
        extremasResultsT,
        reductionKernels,
        curAbsScores,
        result,
      });
    });

    return _.flatten(result);
  }

  protected applyFilter(image: tf.Tensor) {
    const [height, width] = image.shape;

    if (_.isNil(height) || _.isNil(width)) return;

    const kernelKey = `w${width}`;

    if (!this.kernelCaches.applyFilter) this.kernelCaches.applyFilter = {};
    if (!this.kernelCaches.applyFilter[kernelKey]) {
      this.kernelCaches.applyFilter[kernelKey] = Kernels.applyFilter(
        height,
        width
      );
    }

    return tf.tidy(() => {
      if (!this.kernelCaches?.applyFilter?.[kernelKey]) return;

      const [program1, program2] = this.kernelCaches.applyFilter[kernelKey];
      const result1 = runner.compileAndRun(program1, [image]);
      const result2 = runner.compileAndRun(program2, [result1]);

      return result2;
    });
  }

  protected buildExtremas(
    image0: tf.Tensor,
    image1: tf.Tensor,
    image2: tf.Tensor
  ) {
    const [height, width] = image1.shape;

    if (_.isNil(height) || _.isNil(width)) return;

    const kernelKey = `w${width}`;

    if (!this.kernelCaches.buildExtremas) this.kernelCaches.buildExtremas = {};
    if (!this.kernelCaches.buildExtremas[kernelKey])
      this.kernelCaches.buildExtremas[kernelKey] = Kernels.buildExtremas(
        height,
        width
      );

    return tf.tidy(() => {
      if (!this.kernelCaches?.buildExtremas?.[kernelKey]) return;

      const program = this.kernelCaches.buildExtremas[kernelKey];

      const newImage0 = this.downsampleBilinear(image0);
      const newImage2 = this.upsampleBilinear(image2, image1);

      if (!newImage0 || !newImage2) return;

      image0 = newImage0;
      image2 = newImage2;

      return runner.compileAndRun(program, [image0, image1, image2]);
    });
  }

  protected computeLocalization(
    prunedExtremasList: number[][],
    dogPyramidImagesT: tf.Tensor[]
  ) {
    if (!this.kernelCaches.computeLocalization) {
      this.kernelCaches.computeLocalization = Kernels.computeLocalization(
        dogPyramidImagesT,
        prunedExtremasList
      );
    }

    return tf.tidy(() => {
      if (!this.kernelCaches.computeLocalization) return;

      const program = this.kernelCaches.computeLocalization[0];

      const pixels = creators.computeLocalization.createResult(
        program,
        prunedExtremasList,
        dogPyramidImagesT
      );

      const localizedExtremas = callbacks.computeLocalization.computeExtremas(
        prunedExtremasList,
        pixels
      );

      return tf.tensor(
        localizedExtremas,
        [localizedExtremas.length, localizedExtremas[0].length],
        'float32'
      );
    });
  }

  // TODO: maybe can try just using average momentum, instead of histogram method. histogram might be overcomplicated
  protected computeOrientationHistograms(
    prunedExtremasT: tf.Tensor,
    pyramidImagesT: tf.Tensor[][]
  ) {
    if (!this.tensorCaches.orientationHistograms) {
      tf.tidy(() => {
        this.tensorCaches.orientationHistograms = {
          radialPropertiesT:
            creators.computeOrientationHistograms.createOrientationHistogramsCache(),
        };
      });
    }

    if (!this.tensorCaches.orientationHistograms) return;

    const { radialPropertiesT } = this.tensorCaches.orientationHistograms;

    const gaussianImagesT: tf.Tensor[] = _.compact(
      _.map(pyramidImagesT, (pyramidImageT, id) => {
        if (id < 1) return;

        return pyramidImageT[1];
      })
    );

    if (!this.kernelCaches.computeOrientationHistograms)
      this.kernelCaches.computeOrientationHistograms =
        Kernels.computeOrientationHistograms(
          pyramidImagesT,
          prunedExtremasT,
          radialPropertiesT,
          DETECTOR_CONSTANTS.ONE_OVER_2PI
        );

    return tf.tidy(() => {
      if (!this.kernelCaches.computeOrientationHistograms) return;

      const [program1, program2] =
        this.kernelCaches.computeOrientationHistograms;

      const result1 = runner.compileAndRun(program1, [
        ...gaussianImagesT,
        prunedExtremasT,
        radialPropertiesT,
      ]);
      const result2 = runner.compileAndRun(program2, [result1]);

      return result2;
    });
  }

  protected computeExtremaAngles(histograms: tf.Tensor) {
    if (!this.kernelCaches.computeExtremaAngles)
      this.kernelCaches.computeExtremaAngles =
        Kernels.computeExtremaAngles(histograms);

    return tf.tidy(() => {
      if (!this.kernelCaches?.computeExtremaAngles) return;

      const program = this.kernelCaches.computeExtremaAngles;

      return runner.compileAndRun(program, [histograms]);
    });
  }

  protected computeExtremaFreak(
    pyramidImagesT: tf.Tensor[][],
    prunedExtremas: tf.Tensor,
    prunedExtremasAngles: tf.Tensor
  ) {
    if (!this.tensorCaches.computeExtremaFreak)
      tf.tidy(() => {
        const freakPoints = tf.tensor(FREAKPOINTS);

        this.tensorCaches.computeExtremaFreak = {
          freakPointsT: tf.keep(freakPoints),
        };
      });

    if (!this.tensorCaches.computeExtremaFreak) return;

    const { freakPointsT } = this.tensorCaches.computeExtremaFreak;

    const gaussianImagesT: tf.Tensor[] = _.compact(
      _.map(pyramidImagesT, (pyramidImageT, id) => {
        if (id < 1) return;

        return pyramidImageT[1];
      })
    );

    if (!this.kernelCaches.computeExtremaFreak)
      this.kernelCaches.computeExtremaFreak = Kernels.computeExtremaFreak(
        pyramidImagesT,
        prunedExtremas
      );

    return tf.tidy(() => {
      if (!this.kernelCaches.computeExtremaFreak) return;

      const [program] = this.kernelCaches.computeExtremaFreak;

      const result = runner.compileAndRun(program, [
        ...gaussianImagesT,
        prunedExtremas,
        prunedExtremasAngles,
        freakPointsT,
      ]);

      return result;
    });
  }

  protected computeFreakDescriptors(extremaFreaks: tf.Tensor) {
    if (!this.tensorCaches.computeFreakDescriptors) {
      const freakDescriptors =
        creators.computeFreakDescriptors.createFreakDescriptors(extremaFreaks);

      if (!freakDescriptors) return;

      this.tensorCaches.computeFreakDescriptors = {
        positionT: freakDescriptors,
      };
    }

    if (!this.tensorCaches?.computeFreakDescriptors) return;

    const { positionT } = this.tensorCaches.computeFreakDescriptors;

    // encode 8 bits into one number
    // trying to encode 16 bits give wrong result in iOS. may integer precision issue
    const descriptorCount = Math.ceil(
      DETECTOR_CONSTANTS.FREAK_CONPARISON_COUNT / 8
    );

    if (!this.kernelCaches.computeFreakDescriptors)
      this.kernelCaches.computeFreakDescriptors =
        Kernels.computeFreakDescriptors(
          extremaFreaks,
          descriptorCount,
          DETECTOR_CONSTANTS.FREAK_CONPARISON_COUNT
        );

    return tf.tidy(() => {
      if (!this.kernelCaches?.computeFreakDescriptors) return;

      const [program] = this.kernelCaches.computeFreakDescriptors;

      return runner.runWebGlProgram(
        program,
        [extremaFreaks, positionT],
        'int32'
      );
    });
  }

  protected differenceImageBinomial(image1: tf.Tensor, image2: tf.Tensor) {
    return tf.tidy(() => {
      return image1.sub(image2);
    });
  }

  protected upsampleBilinear(image: tf.Tensor, targetImage: tf.Tensor) {
    const [, width] = image.shape;
    const [targetHeight, targeWidth] = targetImage.shape;

    if (_.isNil(width) || _.isNil(targetHeight) || _.isNil(targeWidth)) return;

    const kernelKey = `w${width}`;

    if (!this.kernelCaches.upsampleBilinear)
      this.kernelCaches.upsampleBilinear = {};

    if (!this.kernelCaches.upsampleBilinear[kernelKey])
      this.kernelCaches.upsampleBilinear[kernelKey] = Kernels.upsampleBilinear(
        targetHeight,
        targeWidth
      );

    return tf.tidy(() => {
      if (!this.kernelCaches?.upsampleBilinear?.[kernelKey]) return;

      const program = this.kernelCaches.upsampleBilinear[kernelKey];

      return runner.compileAndRun(program, [image]);
    });
  }

  protected downsampleBilinear(image: tf.Tensor) {
    const [height, width] = image.shape;

    if (_.isNil(height) || _.isNil(width)) return;

    const kernelKey = `w${width}`;

    if (!this.kernelCaches.downsampleBilinear)
      this.kernelCaches.downsampleBilinear = {};

    if (!this.kernelCaches.downsampleBilinear[kernelKey])
      this.kernelCaches.downsampleBilinear[kernelKey] =
        Kernels.downsampleBilinear(height, width);

    return tf.tidy(() => {
      if (!this.kernelCaches?.downsampleBilinear?.[kernelKey]) return;

      const program = this.kernelCaches.downsampleBilinear[kernelKey];

      return runner.compileAndRun(program, [image]);
    });
  }

  protected smoothHistograms(histograms: tf.Tensor) {
    if (!this.kernelCaches.smoothHistograms)
      this.kernelCaches.smoothHistograms = Kernels.smoothHistograms(histograms);

    return tf.tidy(() => {
      const program = this.kernelCaches.smoothHistograms;

      if (!program) return;

      for (
        let i = 0;
        i < DETECTOR_CONSTANTS.ORIENTATION_SMOOTHING_ITERATIONS;
        i++
      )
        histograms = runner.compileAndRun(program, [histograms]);

      return histograms;
    });
  }
}

export default BaseDetector;
