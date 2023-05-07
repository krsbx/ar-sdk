import * as tf from '@tensorflow/tfjs';
import { Constants, Utils } from '@krsbx/ar-sdk-core';
import BaseTracker from './BaseTracker';

const TRACKER_CONSTANTS = Constants.IMAGE_TARGET.TRACKER;
const { computeScreenCoordinate } = Utils.projections.coordinates;
const { buildModelViewProjectionTransform } = Utils.projections.transforms;

class Tracker extends BaseTracker {
  constructor(arg: {
    trackingDataList: ArTracker.TrackingFeature[][];
    projectionTransform: number[][];
    debugMode?: boolean;
  }) {
    super(arg);
  }

  public dummyRun(inputT: tf.Tensor) {
    const transform = [
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ];

    for (
      let targetIndex = 0;
      targetIndex < this.featurePointsListT.length;
      targetIndex++
    ) {
      this.track({
        inputImageT: inputT,
        lastModelViewTransform: transform,
        targetIndex,
      });
    }
  }

  public track(arg: {
    inputImageT: tf.Tensor;
    lastModelViewTransform: number[][];
    targetIndex: number;
  }) {
    const { inputImageT, lastModelViewTransform, targetIndex } = arg;

    const debugExtra: ArTracker.DebugExtra = {} as ArTracker.DebugExtra;

    const modelViewProjectionTransform = buildModelViewProjectionTransform(
      this.projectionTransform,
      lastModelViewTransform
    );

    const modelViewProjectionTransformT = this.buildAdjustedModelViewTransform(
      modelViewProjectionTransform
    );

    const featurePointsT = this.featurePointsListT[targetIndex];
    const imagePixelsT = this.imagePixelsListT[targetIndex];
    const imagePropertiesT = this.imagePropertiesListT[targetIndex];

    const projectedImageT = this.computeProjection(
      modelViewProjectionTransformT,
      inputImageT,
      targetIndex
    );

    if (!projectedImageT) return;

    const matchingResult = this.computeMatching({
      featurePointsT,
      imagePixelsT,
      imagePropertiesT,
      projectedImageT,
    });

    if (!matchingResult) return;

    const { matchingPointsT, simT } = matchingResult;

    const matchingPoints = matchingPointsT.arraySync() as number[][];
    const sim = simT.arraySync() as number[];

    const trackingFrame = this.trackingKeyframeList[targetIndex];
    const { goodTrack, screenCoords, worldCoords } = matchingPoints.reduce(
      (prev, curr, i) => {
        if (
          sim[i] < TRACKER_CONSTANTS.AR2_SIM_THRESH ||
          i > trackingFrame.points.length
        )
          return prev;

        prev.goodTrack.push(i);

        const point = computeScreenCoordinate(
          modelViewProjectionTransform,
          curr[0],
          curr[1]
        );

        prev.screenCoords.push(point);
        prev.worldCoords.push({
          x: trackingFrame.points[i].x / trackingFrame.scale,
          y: trackingFrame.points[i].y / trackingFrame.scale,
          z: 0,
        });

        return prev;
      },
      {
        worldCoords: [] as { x: number; y: number; z: number }[],
        screenCoords: [] as { x: number; y: number }[],
        goodTrack: [] as number[],
      }
    );

    if (this.debugMode) {
      Object.assign(debugExtra, {
        projectedImage: projectedImageT.arraySync() as number[][],
        matchingPoints,
        trackedPoints: screenCoords,
        goodTrack,
      });
    }

    // tensors cleanup
    modelViewProjectionTransformT.dispose();
    projectedImageT.dispose();
    matchingPointsT.dispose();
    simT.dispose();

    return { worldCoords, screenCoords, debugExtra };
  }
}

export default Tracker;
