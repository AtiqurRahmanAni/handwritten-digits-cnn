import { BarChart } from "@mui/x-charts";
import ImageFromArray from "./ImageFromArray";

const xLabels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

const Visualize = ({ data }) => {
  const { probas, predicted, activations } = data;

  return (
    <div className="max-h-dvh items-center overflow-y-scroll py-2 scrollbar-custom">
      <div className="flex flex-col items-center">
        <BarChart
          width={450}
          height={300}
          series={[{ data: probas, label: "Probability", id: "probasId" }]}
          yAxis={[{ data: xLabels, scaleType: "band", label: "Digits" }]}
          xAxis={[{ label: "Probability" }]}
          layout="horizontal"
        />
        <div className="text-center font-semibold text-lg">
          <p>{`Predicted digit: ${predicted}`}</p>
        </div>
      </div>
      <hr className="h-[2px] bg-gray-400" />
      <div className="mt-4">
        {["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2"].map(
          (layer) => (
            <div key={layer}>
              <div className="text-center font-semibold text-xl">
                <p>{`Activations of ${layer}`}</p>
              </div>
              <div className="flex flex-wrap">
                {activations[layer].map((activation, idx) => (
                  <ImageFromArray key={idx} pixelData={activation} />
                ))}
              </div>
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default Visualize;
