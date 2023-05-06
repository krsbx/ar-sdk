import { Tensor } from '@tensorflow/tfjs';
import Detector from './Detector';

class CropDetector {
  private width: number;
  private height: number;
  private debugMode: boolean;
  private cropSize: number;
  private lastRandomIndex: number;
  private detector: Detector;

  constructor(arg: { width: number; height: number; debugMode?: boolean }) {
    this.width = arg.width;
    this.height = arg.height;
    this.debugMode = arg.debugMode ?? false;

    this.cropSize = this.getCropSize(arg.width, arg.height);
    this.detector = new Detector(arg);

    this.lastRandomIndex = 4;
  }

  private getCropSize(width: number, height: number) {
    // nearest power of 2, min dimensions
    const minDimension = Math.min(width, height) / 2;
    const cropSize = 2 ** Math.round(Math.log(minDimension) / Math.log(2));

    return cropSize;
  }

  private _detect(inputImageT: Tensor, startX: number, startY: number) {
    const cropInputImageT = inputImageT.slice(
      [startY, startX],
      [this.cropSize, this.cropSize]
    );

    const detectionResult = this.detector.detect(cropInputImageT);

    if (!detectionResult) return;

    const { featurePoints, debugExtra } = detectionResult;

    featurePoints.forEach((p) => {
      p.x += startX;
      p.y += startY;
    });

    if (this.debugMode)
      debugExtra.projectedImage = cropInputImageT.arraySync() as number[][];

    cropInputImageT.dispose();

    return {
      featurePoints,
      debugExtra,
    };
  }

  public detect(inputImageT: Tensor) {
    const startY = Math.floor(this.height / 2 - this.cropSize / 2);
    const startX = Math.floor(this.width / 2 - this.cropSize / 2);
    const result = this._detect(inputImageT, startX, startY);

    if (!result) return;

    if (this.debugMode)
      result.debugExtra.crop = {
        startX,
        startY,
        cropSize: this.cropSize,
      };

    return result;
  }

  public detectMoving(inputImageT: Tensor) {
    // loop a few locations around center
    const dx = this.lastRandomIndex % 3;
    const dy = Math.floor(this.lastRandomIndex / 3);

    let startY = Math.floor(
      this.height / 2 - this.cropSize + (dy * this.cropSize) / 2
    );
    let startX = Math.floor(
      this.width / 2 - this.cropSize + (dx * this.cropSize) / 2
    );

    if (startX < 0) startX = 0;
    if (startY < 0) startY = 0;
    if (startX >= this.width - this.cropSize)
      startX = this.width - this.cropSize - 1;
    if (startY >= this.height - this.cropSize)
      startY = this.height - this.cropSize - 1;

    this.lastRandomIndex = (this.lastRandomIndex + 1) % 9;

    const result = this._detect(inputImageT, startX, startY);

    return result;
  }
}

export default CropDetector;
