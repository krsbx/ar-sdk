import * as tf from '@tensorflow/tfjs';
import _ from 'lodash';
import { Constants, Kernels } from '@krsbx/ar-sdk-core';
import { runner } from '../utils';

const TRACKER_CONSTANTS = Constants.IMAGE_TARGET.TRACKER;

class BaseTracker {
  protected projectionTransform: number[][];
  protected trackingKeyframeList: ArTracker.TrackingFeature[];
  protected debugMode: boolean;
  protected kernelCaches: {
    computeMatching?: tf.GPGPUProgram[];
    computeProjection?: Record<string, tf.GPGPUProgram>;
  };
  protected featurePointsListT: tf.Tensor[];
  protected imagePixelsListT: tf.Tensor[];
  protected imagePropertiesListT: tf.Tensor[];
  protected templateOneSize: number;
  protected templateSize: number;
  protected templateGap: number;
  protected searchOneSize: number;
  protected searchGap: number;
  protected searchSize: number;

  constructor(arg: {
    trackingDataList: ArTracker.TrackingFeature[][];
    projectionTransform: number[][];
    debugMode?: boolean;
  }) {
    this.projectionTransform = arg.projectionTransform;
    this.trackingKeyframeList = arg.trackingDataList.map(
      (trackingData) => trackingData[TRACKER_CONSTANTS.TRACKING_KEYFRAME]
    );
    this.debugMode = arg?.debugMode ?? false;

    // prebuild feature and marker pixel tensors
    const maxCount = Math.max(
      ...this.trackingKeyframeList.map(({ points }) => points.length)
    );

    const { featurePointListT, imagePixelsListT, imagePropertiesListT } =
      this.trackingKeyframeList.reduce(
        (prev, curr) => {
          const { featurePoints, imagePixels, imageProperties } =
            runner.preBuild(curr, maxCount);

          prev.featurePointListT.push(featurePoints);
          prev.imagePixelsListT.push(imagePixels);
          prev.imagePropertiesListT.push(imageProperties);

          return prev;
        },
        {
          featurePointListT: [] as tf.Tensor[],
          imagePixelsListT: [] as tf.Tensor[],
          imagePropertiesListT: [] as tf.Tensor[],
        }
      );

    this.featurePointsListT = featurePointListT;
    this.imagePixelsListT = imagePixelsListT;
    this.imagePropertiesListT = imagePropertiesListT;

    this.templateOneSize = TRACKER_CONSTANTS.AR2_DEFAULT_TS;
    this.templateSize = this.templateOneSize * 2 + 1;
    this.templateGap = TRACKER_CONSTANTS.AR2_DEFAULT_TS_GAP;
    this.searchOneSize = TRACKER_CONSTANTS.AR2_SEARCH_SIZE * this.templateGap;
    this.searchGap = TRACKER_CONSTANTS.AR2_SEARCH_GAP;
    this.searchSize = this.searchOneSize * 2 + 1;

    this.kernelCaches = {};
  }

  protected computeMatching(arg: {
    featurePointsT: tf.Tensor;
    imagePixelsT: tf.Tensor;
    imagePropertiesT: tf.Tensor;
    projectedImageT: tf.Tensor;
  }) {
    const { featurePointsT, imagePixelsT, imagePropertiesT, projectedImageT } =
      arg;

    const [height, width] = projectedImageT.shape;
    const [count] = featurePointsT.shape;

    if (_.isNil(height) || _.isNil(width) || _.isNil(count)) return;

    if (!this.kernelCaches.computeMatching)
      this.kernelCaches.computeMatching = Kernels.tracker.computeMatching({
        templateOneSize: this.templateOneSize,
        templateSize: this.templateSize,
        searchOneSize: this.searchOneSize,
        searchGap: this.searchGap,
        searchSize: this.searchSize,
        targetHeight: height,
        targetWidth: width,
        featureCount: count,
      });

    return tf.tidy(() => {
      if (!this.kernelCaches.computeMatching) return;

      const programs = this.kernelCaches.computeMatching;

      const allSims = runner.compileAndRun(programs[0], [
        featurePointsT,
        imagePixelsT,
        imagePropertiesT,
        projectedImageT,
      ]);
      const maxIndex = allSims.argMax(1);

      const matchingPointsT = runner.compileAndRun(programs[1], [
        featurePointsT,
        imagePropertiesT,
        maxIndex,
      ]);

      const simT = runner.compileAndRun(programs[2], [allSims, maxIndex]);

      return { matchingPointsT, simT };
    });
  }

  protected computeProjection(
    modelViewProjectionTransformT: tf.Tensor,
    inputImageT: tf.Tensor,
    targetIndex: number
  ) {
    const markerWidth = this.trackingKeyframeList[targetIndex].width;
    const markerHeight = this.trackingKeyframeList[targetIndex].height;
    const markerScale = this.trackingKeyframeList[targetIndex].scale;
    const kernelKey = `${markerWidth}-${markerHeight}-${markerScale}`;

    if (!this.kernelCaches.computeProjection)
      this.kernelCaches.computeProjection = {};

    if (!this.kernelCaches.computeProjection[kernelKey]) {
      this.kernelCaches.computeProjection[kernelKey] =
        Kernels.tracker.computeProjection(
          markerHeight,
          markerWidth,
          markerScale
        );
    }

    return tf.tidy(() => {
      if (!this.kernelCaches?.computeProjection?.[kernelKey]) return;

      const program = this.kernelCaches.computeProjection[kernelKey];
      const result = runner.compileAndRun(program, [
        modelViewProjectionTransformT,
        inputImageT,
      ]);

      return result;
    });
  }

  protected buildAdjustedModelViewTransform(
    modelViewProjectionTransform: number[][]
  ) {
    return tf.tidy(() => {
      const modelViewProjectionTransformAdjusted: number[][] =
        modelViewProjectionTransform.map((row) =>
          row.map((c) => c / TRACKER_CONSTANTS.PRECISION_ADJUST)
        );

      const t = tf.tensor(modelViewProjectionTransformAdjusted, [3, 4]);

      return t;
    });
  }
}

export default BaseTracker;
