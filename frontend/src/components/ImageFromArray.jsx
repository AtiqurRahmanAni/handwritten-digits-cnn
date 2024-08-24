import React, { useEffect, useRef } from "react";

const ImageFromArray = ({ pixelData }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const canvasWidth = 28;
    const canvasHeight = 28;

    // Set canvas size
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Create an ImageData object for the fixed canvas size
    const imageData = ctx.createImageData(canvasWidth, canvasHeight);

    // Flatten the 2D array and populate the ImageData object
    const data = new Uint8ClampedArray(canvasWidth * canvasHeight * 4); // RGBA

    // Calculate scaling factors
    const scaleX = canvasWidth / pixelData[0].length;
    const scaleY = canvasHeight / pixelData.length;

    for (let y = 0; y < canvasHeight; y++) {
      for (let x = 0; x < canvasWidth; x++) {
        // Scale the coordinates
        const srcX = Math.floor(x / scaleX);
        const srcY = Math.floor(y / scaleY);

        // Ensure the source coordinates are within bounds
        const clampedSrcX = Math.min(srcX, pixelData[0].length - 1);
        const clampedSrcY = Math.min(srcY, pixelData.length - 1);

        const index = (y * canvasWidth + x) * 4;
        const pixelValue = pixelData[clampedSrcY][clampedSrcX];

        // Set the RGBA values; here, R, G, and B are the pixelValue, and A is 255 (fully opaque)
        data[index] = pixelValue; // Red
        data[index + 1] = pixelValue; // Green
        data[index + 2] = pixelValue; // Blue
        data[index + 3] = 255; // Alpha
      }
    }

    imageData.data.set(data);
    ctx.putImageData(imageData, 0, 0);
  }, [pixelData]);

  return (
    <canvas
      ref={canvasRef}
      width={28}
      height={28}
      style={{ border: "1px solid black" }}
    />
  );
};

export default ImageFromArray;
