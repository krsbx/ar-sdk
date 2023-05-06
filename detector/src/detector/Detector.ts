import _ from 'lodash';
import * as tf from '@tensorflow/tfjs';
import { Constants } from '@krsbx/ar-sdk-core';
import BaseDetector from './BaseDetector';

const DETECTOR_CONSTANTS = Constants.IMAGE_TARGET.DETECTOR;

class Detector extends BaseDetector {
  constructor(arg: { width: number; height: number; debugMode?: boolean }) {
    super(arg);
  }

  // Build gaussian pyramid images, two images per octave
  private buildPyramidImage(inputImageT: tf.Tensor) {
    const pyramidImagesT: tf.Tensor[][] = [];

    for (let i = 0; i < this.numOctaves; i++) {
      const image1T =
        i === 0
          ? this.applyFilter(inputImageT)
          : this.downsampleBilinear(
              pyramidImagesT[i - 1][pyramidImagesT[i - 1].length - 1]
            );

      if (!image1T) continue;

      const image2T = this.applyFilter(image1T);

      if (!image2T) continue;

      pyramidImagesT.push([image1T, image2T]);
    }

    return pyramidImagesT;
  }

  // Build difference-of-gaussian (dog) pyramid
  private buildDogPyramid(pyramidImagesT: tf.Tensor[][]) {
    const dogPyramidImagesT: tf.Tensor[] = [];

    for (let i = 0; i < this.numOctaves; i++) {
      const dogImageT = this.differenceImageBinomial(
        pyramidImagesT[i][0],
        pyramidImagesT[i][1]
      );

      dogPyramidImagesT.push(dogImageT);
    }

    return dogPyramidImagesT;
  }

  // find local maximum/minimum
  private getExtremas(dogPyramidImagesT: tf.Tensor[]) {
    const extremasResultsT: tf.Tensor[] = [];

    for (let i = 1; i < this.numOctaves - 1; i++) {
      const extremasResultT = this.buildExtremas(
        dogPyramidImagesT[i - 1],
        dogPyramidImagesT[i],
        dogPyramidImagesT[i + 1]
      );

      if (!extremasResultT) continue;

      extremasResultsT.push(extremasResultT);
    }

    return extremasResultsT;
  }

  // get featured points from the image
  private getFeaturePoints(
    prunedExtremasArr: number[][],
    freakDescriptorsArr: number[][],
    extremaAnglesArr: number[]
  ) {
    const featurePoints: ArDetector.MaximaMinimaPoint[] = _(prunedExtremasArr)
      .map((prunedExtremas, i) => {
        if (prunedExtremas[0] === 0) return;

        const descriptors: number[] = [];

        for (let m = 0; m < freakDescriptorsArr[i].length; m += 4) {
          const v1 = freakDescriptorsArr[i][m];
          const v2 = freakDescriptorsArr[i][m + 1];
          const v3 = freakDescriptorsArr[i][m + 2];
          const v4 = freakDescriptorsArr[i][m + 3];

          const combined =
            v1 * DETECTOR_CONSTANTS.EIGHT_BIT_COLOR ** 3 +
            v2 * DETECTOR_CONSTANTS.EIGHT_BIT_COLOR ** 2 +
            v3 * DETECTOR_CONSTANTS.EIGHT_BIT_COLOR +
            v4;

          descriptors.push(combined);
        }

        const octave = prunedExtremas[1];
        const y = prunedExtremas[2];
        const x = prunedExtremas[3];
        const originalX = x * 2 ** octave + 2 ** (octave - 1) - 0.5;
        const originalY = y * 2 ** octave + 2 ** (octave - 1) - 0.5;
        const scale = 2 ** octave;

        return {
          maxima: prunedExtremas[0] > 0,
          x: originalX,
          y: originalY,
          scale,
          angle: extremaAnglesArr[i],
          descriptors,
        };
      })
      .compact()
      .value();

    return featurePoints;
  }

  public detect(inputImageT: tf.Tensor) {
    const debugExtra: ArDetector.DebugExtra = {} as ArDetector.DebugExtra;

    const pyramidImagesT = this.buildPyramidImage(inputImageT);
    const dogPyramidImagesT = this.buildDogPyramid(pyramidImagesT);
    const extremasResultsT = this.getExtremas(dogPyramidImagesT);

    // divide the input into N by N buckets, and for each bucket,
    // collect the top 5 most significant extrema across extremas in all scale level
    // result would be NUM_BUCKETS x NUM_FEATURES_PER_BUCKET extremas
    const prunedExtremasList = this.applyPrune(extremasResultsT);
    if (!prunedExtremasList) return;

    const prunedExtremasT = this.computeLocalization(
      prunedExtremasList,
      dogPyramidImagesT
    );

    if (!prunedExtremasT) return;

    // compute the orientation angle for each pruned extremas
    const extremaHistogramsT = this.computeOrientationHistograms(
      prunedExtremasT,
      pyramidImagesT
    );

    if (!extremaHistogramsT) return;

    const smoothedHistogramsT = this.smoothHistograms(extremaHistogramsT);

    if (!smoothedHistogramsT) return;

    const extremaAnglesT = this.computeExtremaAngles(smoothedHistogramsT);

    if (!extremaAnglesT) return;

    // to compute freak descriptors, we first find the pixel value of 37 freak points for each extrema
    const extremaFreaksT = this.computeExtremaFreak(
      pyramidImagesT,
      prunedExtremasT,
      extremaAnglesT
    );

    if (!extremaFreaksT) return;

    // compute the binary descriptors
    const freakDescriptorsT = this.computeFreakDescriptors(extremaFreaksT);

    if (!freakDescriptorsT) return;

    const prunedExtremasArr = prunedExtremasT.arraySync() as number[][];
    const extremaAnglesArr = extremaAnglesT.arraySync() as number[];
    const freakDescriptorsArr = freakDescriptorsT.arraySync() as number[][];

    if (this.debugMode) {
      Object.assign(debugExtra, {
        pyramidImages: pyramidImagesT.map((ts) =>
          ts.map((t) => t.arraySync())
        ) as number[][],
        dogPyramidImages: dogPyramidImagesT.map(
          (t) => (t?.arraySync() as number[]) ?? null
        ),
        extremasResults: extremasResultsT.map((t) => t.arraySync()) as number[],
        extremaAngles: extremaAnglesT.arraySync() as number[],
        prunedExtremas: prunedExtremasList,
        localizedExtremas: prunedExtremasT.arraySync() as number[][],
      });
    }

    // Cleanup tensors
    pyramidImagesT.forEach((ts) => ts.forEach((t) => t?.dispose()));
    dogPyramidImagesT.forEach((t) => t?.dispose());
    extremasResultsT.forEach((t) => t.dispose());
    prunedExtremasT.dispose();
    extremaHistogramsT.dispose();
    smoothedHistogramsT.dispose();
    extremaAnglesT.dispose();
    extremaFreaksT.dispose();
    freakDescriptorsT.dispose();

    const featurePoints: ArDetector.MaximaMinimaPoint[] = this.getFeaturePoints(
      prunedExtremasArr,
      freakDescriptorsArr,
      extremaAnglesArr
    );

    return {
      featurePoints,
      debugExtra,
    };
  }

  public detectImageData(imageData: number[]) {
    const arr = new Uint8ClampedArray(4 * imageData.length);
    for (let i = 0; i < imageData.length; i++) {
      arr[4 * i] = imageData[i];
      arr[4 * i + 1] = imageData[i];
      arr[4 * i + 2] = imageData[i];
      arr[4 * i + 3] = 255;
    }

    const img = new ImageData(arr, this.width, this.height);

    return this.detect(img as never);
  }
}

export default Detector;
