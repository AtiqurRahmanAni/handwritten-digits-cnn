import "./App.css";
import Canvas from "./components/Canvas";
import axiosInstance from "./utils/axiosInstance";
import Visualize from "./components/Visualize";
import { useState } from "react";
import { toast, Toaster } from "react-hot-toast";

const App = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const classify = async (image) => {
    try {
      setLoading(true);
      const response = await axiosInstance.post("/api/classify", { image });
      setData(response.data);
    } catch (err) {
      if (err?.response) {
        toast.error(err.response.data.error);
      } else {
        toast.error("Something went wrong");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="flex gap-x-8 px-4">
        <div className="min-h-dvh flex items-center">
          <Canvas classify={classify} loading={loading} />
        </div>
        {data ? (
          <div>
            <Visualize data={data} />
          </div>
        ) : (
          <div className="text-2xl font-semibold w-full min-h-dvh flex justify-center items-center">
            <h1>Draw a digit to visualize</h1>
          </div>
        )}
      </div>
      <Toaster position="bottom-right" reverseOrder={false} />
    </>
  );
};

export default App;
