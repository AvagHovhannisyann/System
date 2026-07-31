import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone server bundle so the Docker runtime image (P1.3) can copy
  // .next/standalone instead of node_modules.
  output: "standalone",
};

export default nextConfig;
