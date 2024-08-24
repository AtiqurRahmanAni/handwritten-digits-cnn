import React, { useEffect, useRef, useState } from "react";
import ModelPreview from "./ModelPreview";

const Canvas = ({ classify, loading = false }) => {
  const CANVAS_SIZE = 200;
  const canvasRef = useRef(null);
  const canvasContextRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    canvas.style.width = `${CANVAS_SIZE}px`;
    canvas.style.height = `${CANVAS_SIZE}px`;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    canvasContextRef.current = ctx;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "black";
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";
    ctx.lineWidth = 8;
    ctx.scale(1, 1);

    document.body.addEventListener("mouseout", bodyMouseOut);

    return () => {
      document.body.removeEventListener("mouseout", bodyMouseOut);
    };
  }, []);

  const bodyMouseOut = (event) => {
    if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
      setIsDrawing(false);
    }
  };

  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    canvasContextRef.current.beginPath();
    canvasContextRef.current.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const finishDrawing = (event) => {
    canvasContextRef.current.closePath();
    setIsDrawing(false);
  };

  const draw = ({ nativeEvent }) => {
    if (isDrawing) {
      const { offsetX, offsetY } = nativeEvent;
      canvasContextRef.current.lineTo(offsetX, offsetY);
      canvasContextRef.current.stroke();
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the entire canvas

    // Reset the background to black after clearing
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const onSubmit = () => {
    const imgData = canvasContextRef.current.getImageData(
      0,
      0,
      CANVAS_SIZE,
      CANVAS_SIZE
    );

    const grayscaleArray2D = [];

    for (let y = 0; y < imgData.height; y++) {
      const row = [];
      for (let x = 0; x < imgData.width; x++) {
        const index = (y * imgData.width + x) * 4;
        const red = imgData.data[index];
        const green = imgData.data[index + 1];
        const blue = imgData.data[index + 2];

        // Calculate the grayscale value using the luminosity method
        const grayscale = 0.299 * red + 0.587 * green + 0.114 * blue;
        row.push(grayscale / 255);
      }

      grayscaleArray2D.push(row);
    }

    classify(grayscaleArray2D);
  };

  return (
    <>
      <div>
        <div className="text-center font-semibold">
          <h3>Draw digit here (one digit)</h3>
        </div>
        <canvas
          className="shadow-sm w-[250px] h-[250px] border-2"
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseUp={finishDrawing}
          onMouseMove={draw}
        />
        <div className="mt-2 flex justify-center gap-x-2">
          <button
            className="bg-red-500 px-3 py-1.5 text-white rounded-md font-semibold transition-colors duration-100 hover:bg-red-600"
            onClick={clearCanvas}
          >
            Clear
          </button>
          <button
            className={`bg-indigo-500 px-3 py-1.5 text-white rounded-md font-semibold transition-colors duration-100 hover:bg-indigo-600 ${
              loading ? "bg-indigo-500/65 pointer-events-none" : ""
            }`}
            onClick={onSubmit}
            disabled={loading}
          >
            {!loading ? "Submit" : "Submitting..."}
          </button>
        </div>
        <div className="text-center mt-2">
          <button className="underline" onClick={() => setIsOpen(true)}>
            View The model
          </button>
        </div>
      </div>
      <ModelPreview isOpen={isOpen} setIsOpen={setIsOpen} />
    </>
  );
};

export default Canvas;
