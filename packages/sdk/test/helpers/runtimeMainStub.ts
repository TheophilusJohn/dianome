// Stands in for the `dianome-runtime` main entry in the node test suite: run() must never import it without WebGPU.
throw new Error("dianome-runtime main entry imported in node (run() must plan server without WebGPU and never import the runtime)");
export {};
