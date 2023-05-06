export function generateVariableName<T>(pyramidImagesT: T[]) {
  const imageVariableNames: string[] = [];

  for (let i = 1; i < pyramidImagesT.length; i++) {
    imageVariableNames.push(`image${i}`);
  }

  return imageVariableNames;
}

export function generateSubCodes<T>(pyramidImagesT: T[]) {
  const subcodes = ['float getPixel(int octave, int y, int x) {'];

  for (let i = 1; i < pyramidImagesT.length; i++) {
    subcodes.push(
      `
      if (octave == ${i}) {
        return getImage${i}(y, x);
      }
      `
    );
  }

  subcodes.push('}');

  return subcodes.join('\n');
}
