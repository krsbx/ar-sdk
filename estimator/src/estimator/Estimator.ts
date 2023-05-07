import { estimate } from './estimate';
import { refineEstimate } from './refineEstimate';

class Estimator {
  private projectionTransform: number[][];

  constructor(projectionTransform: number[][]) {
    this.projectionTransform = projectionTransform;
  }

  // Solve homography between screen points and world points using Direct Linear Transformation
  // then decompose homography into rotation and translation matrix (i.e. modelViewTransform)
  public estimate({
    screenCoords,
    worldCoords,
  }: {
    screenCoords: ArEstimator.Vector2[];
    worldCoords: ArEstimator.Vector3[];
  }) {
    const modelViewTransform = estimate({
      screenCoords,
      worldCoords,
      projectionTransform: this.projectionTransform,
    });

    return modelViewTransform;
  }

  // Given an initial guess of the modelViewTransform and new pairs of screen-world coordinates,
  // use Iterative Closest Point to refine the transformation
  // refineEstimate({initialModelViewTransform, screenCoords, worldCoords}) {
  public refineEstimate(arg: {
    initialModelViewTransform: number[][];
    worldCoords: ArEstimator.Vector3[];
    screenCoords: ArEstimator.Vector2[];
  }) {
    const { initialModelViewTransform, worldCoords, screenCoords } = arg;

    const updatedModelViewTransform = refineEstimate({
      projectionTransform: this.projectionTransform,
      initialModelViewTransform,
      worldCoords,
      screenCoords,
    });

    return updatedModelViewTransform;
  }
}

export default Estimator;
