import imageTarget from './image-target';

export const AR_STATE = {
  RENDER_START: 'render-start',
  AR_ERROR: 'ar-error',
  AR_READY: 'ar-ready',
} as const;

export default { IMAGE_TARGET: imageTarget, AR_STATE };
