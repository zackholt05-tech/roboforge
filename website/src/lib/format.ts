// Display formatting. Values are formatted, never computed: every number shown on the site is a
// field of a frozen results file (findings.json) or of a recorded episode.

export const FAMILY: Record<string, string> = {
  bc_mlp: "BC-MLP",
  bc_rnn: "BC-RNN",
  bc_mlp_chunk: "BC-MLP-chunk",
  diffusion_unet: "Diffusion U-Net",
  random_uniform: "Random actions",
  rgb_bc_chunk: "Visual BC-chunk",
  rgb_diffusion: "Visual diffusion",
  pc_diffusion: "Point-cloud diffusion",
};

export const FAMILY_NOTE: Record<string, string> = {
  bc_mlp: "one-step MLP from the current state",
  bc_rnn: "LSTM over a 10-step history, one action per step",
  bc_mlp_chunk: "MLP predicting a 16-step action chunk, 8 executed",
  diffusion_unet: "conditional 1D U-Net denoiser, DDIM-10, 16-step chunks, 8 executed",
  random_uniform: "non-learned sanity baseline, uniform random actions",
  rgb_diffusion: "two 84² cameras + proprioception, diffusion head",
  rgb_bc_chunk: "two 84² cameras + proprioception, chunked BC head",
  pc_diffusion: "512-point cloud from depth + proprioception (DP3-inspired)",
};

export const TASK: Record<string, string> = { lift: "Lift", can: "Can" };

export const CATEGORY: Record<string, string> = {
  success: "Success",
  engineering_error: "Engineering error",
  gripper_never_closed: "Gripper never closed",
  no_grasp_drift: "No grasp, drifted away",
  no_grasp_near: "No grasp, stayed near the object",
  timeout_holding: "Held the object at the time limit",
  in_goal_bin_not_success: "In the goal bin, not counted as success",
  dropped: "Grasped, then dropped",
};

export const PARTITION: Record<string, string> = {
  eval_final_test: "held-out test start",
  eval_validation: "validation start",
  eval_robustness: "robustness set start",
};

export const pct = (x: number, d = 1): string => `${(100 * x).toFixed(d)} %`;
export const pp = (x: number, d = 1): string => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(d)}`;
export const ciPct = ([a, b]: [number, number], d = 1): string =>
  `[${(100 * a).toFixed(d)}, ${(100 * b).toFixed(d)}]`;
export const ciPp = ([a, b]: [number, number], d = 1): string => `[${pp(a, d)}, ${pp(b, d)}]`;
export const short = (h: string | null | undefined, n = 12): string => (h ? h.slice(0, n) : "none");
export const seconds = (s: number): string => `${s.toFixed(2)} s`;
export const sepLabel = (s: string): string => s.replace("separated (+)", "separated, positive")
  .replace("separated (-)", "separated, negative");
