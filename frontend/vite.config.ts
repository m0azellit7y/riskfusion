import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8000; the browser talks to /api and Vite forwards it.
const proxy = {
  "/api": {
    target: process.env.RISKFUSION_API_URL ?? "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (p: string) => p.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
});
