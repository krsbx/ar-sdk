import kernel1 from './kernel1';
import kernel2 from './kernel2';
import kernel3 from './kernel3';

const computeMatching = (arg: {
  templateOneSize: number;
  templateSize: number;
  searchOneSize: number;
  searchGap: number;
  searchSize: number;
  targetHeight: number;
  targetWidth: number;
  featureCount: number;
}) => {
  const KERNEL1 = kernel1(arg);
  const KERNEL2 = kernel2(arg);
  const KERNEL3 = kernel3(arg.featureCount);

  return [KERNEL1, KERNEL2, KERNEL3];
};

export default computeMatching;
