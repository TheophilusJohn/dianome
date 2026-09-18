// `dianome-runtime/planner`: the pure planner (no WebGPU, no kernels). The SDK's run() imports this entry to plan;
// the main entry (Runtime, kernels) only when the plan puts blocks on this device.
export { plan, feasibleN, serverShare, lmHeadBlockEquivalents, cost0Per1M, DEFAULT_GPU_BUDGET } from "../planner";
export type { Mode, PlanInput, Plan, PlanCandidate, PlanBreakdown, PlanModel, PlanDevice, PlanNetwork, PlanServer, PlanRate, PlanPolicy, PlanPrompt, PrivacyBand, PrivacyRow } from "../planner";
