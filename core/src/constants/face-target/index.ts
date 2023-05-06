export const DEFAULT_FILTER_CUTOFF = 0.001; // 1Hz. time period in milliseconds
export const DEFAULT_FILTER_BETA = 1;

export const AR_COMPONENT_NAME = {
  FACE: 'arsdk-face',
  FACE_TARGET: 'arsdk-face-target',
  FACE_SYSTEM: 'arsdk-face-system',
  DEFAULT_OCCLUDER: 'arsdk-face-default-face-occluder',
  OCCULDER: 'arsdk-face-occluder',
};

export const AR_EVENT_NAME = {
  MODEL_LOADED: 'model-loaded',
  MODEL_ERROR: 'model-error',
  TARGET_FOUND: 'face-targetFound',
  TARGET_LOST: 'face-targetLost',
};
