import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  preview: {
    port: 5173,
  },
  server: {
    port: 5173,
  },
  plugins: [react()],
});
