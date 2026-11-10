// Runtime schemas of every Observatory data file (MASTER_SPEC §8.5). Each file the site reads is
// parsed here before use; a file that does not match is refused with an explicit error, never
// patched. The Python side is scripts/export_observatory.py and src/roboforge/viz/obs_*.py.
import { z } from "zod";

const vec3 = z.tuple([z.number(), z.number(), z.number()]);
const quat = z.tuple([z.number(), z.number(), z.number(), z.number()]);

export const Outcome = z.object({
  success: z.boolean(),
  first_success_step: z.number().int().nullable(),
  steps: z.number().int(),
  stop_reason: z.string(),
  terminated: z.boolean(),
  truncated: z.boolean(),
});

export const Geom = z.object({
  body: z.string(),
  kind: z.enum(["sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]),
  name: z.string().nullable(),
  size: vec3,
  pos: vec3,
  quat_wxyz: quat,
  rgba: quat,
  role: z.enum(["robot_light", "robot_dark", "robot_accent", "pedestal", "object", "table",
    "fixture", "target"]),
  mesh: z.string().optional(),
});

export const MeshSummary = z.object({
  name: z.string(),
  n_vert: z.number().int(),
  n_face: z.number().int(),
  face_sha256: z.string().length(64),
});

export const CameraPose = z.object({ pos: vec3, quat_wxyz: quat, fovy_deg: z.number() });

export const ReferenceFrame = z.object({
  frame: z.number().int(),
  camera: z.string(),
  size: z.number().int(),
  rgb: z.string(),
  mask: z.string(),
  pixels: z.object({ robot: z.number().int(), object: z.number().int() }),
  camera_pose: CameraPose,
});

export const Joint = z.object({
  name: z.string(),
  type: z.enum(["free", "ball", "slide", "hinge"]),
  axis: vec3,
  pos: vec3,
  qpos0: z.array(z.number()),
});

export const HierarchyBody = z.object({
  name: z.string(),
  parent: z.string().nullable(),
  category: z.string(),
  local_pos: vec3,
  local_quat_wxyz: quat,
  joints: z.array(Joint),
});

export const Clip = z.object({
  schema: z.literal("roboforge.observatory_clip/1"),
  id: z.string(),
  label: z.string(),
  kind: z.enum(["level1_heldout", "validation_denoising", "level2_placement"]),
  roles: z.array(z.string()),
  task: z.enum(["lift", "can"]),
  family: z.string(),
  seed: z.number().int(),
  state_id: z.string(),
  partition: z.string(),
  suite: z.string().nullable(),
  policy: z.object({ label: z.string(), name: z.string(), kind: z.string(),
    autonomous: z.boolean(), learned: z.boolean() }),
  checkpoint: z.object({ sha256: z.string().nullable(), step: z.number().int().nullable(),
    train_run_id: z.string().nullable() }),
  outcome: Outcome,
  provenance: z.object({
    eval_run_id: z.string(),
    record: z.string(),
    physics_sha256: z.string().length(64),
    arrays_sha256: z.string().length(64),
    platform: z.string(),
    eval_git: z.object({ sha: z.string(), dirty: z.boolean() }).passthrough().nullable(),
    simulator: z.object({ engine: z.string(), control_dt_s: z.number(),
      packages: z.record(z.string(), z.string()) }).passthrough(),
    initial_state: z.object({ state_sha256: z.string(), model_xml_sha256: z.string() })
      .passthrough(),
    frame_convention: z.string(),
    quaternion_convention: z.string(),
  }),
  control_dt_s: z.number(),
  bodies: z.array(z.object({ name: z.string(), category: z.string(), dynamic: z.boolean() })),
  eef_body: z.string(),
  geoms: z.array(Geom),
  meshes: z.record(z.string(), MeshSummary.passthrough()),
  cameras: z.record(z.string(), CameraPose),
  frames: z.object({
    n: z.number().int().positive(),
    time_s: z.array(z.number()),
    success: z.array(z.boolean()),
    static_pos: z.array(vec3),
    static_quat_wxyz: z.array(quat),
    pos: z.array(z.array(vec3)),
    quat_wxyz: z.array(z.array(quat)),
    dynamic_index: z.array(z.number().int()),
    joint_qpos: z.array(z.array(z.number())),
  }),
  joints: z.array(z.string()),
  hierarchy: z.array(HierarchyBody),
  success_rule: z.object({ text: z.string(), source: z.string(),
    z_threshold_m: z.number().optional(), table_top_z_m: z.number().optional() }).passthrough(),
  actions: z.object({
    names: z.array(z.string()),
    executed: z.array(z.array(z.number())),
    clipped: z.array(z.boolean()),
    decision_index: z.array(z.number().int()),
    chunk_offset: z.array(z.number().int()),
    latency_ms: z.array(z.number().nullable()),
  }),
  media: z.object({ video: z.object({ file: z.string(), camera: z.string(), fps: z.number(),
    n_frames: z.number().int(), size: z.number().int(), label: z.string() }) }),
  reference_frames: z.array(ReferenceFrame).optional(),
  policy_views: z.object({ frames: z.array(z.number().int()), cameras: z.array(z.string()),
    size: z.number().int(), files: z.record(z.string(), z.array(z.string())),
    label: z.string() }).optional(),
  clouds: z.string().optional(),
  denoise: z.string().optional(),
  verification: z.object({
    recomputed_vs_recorded: z.object({ body_pos_max_abs_m: z.number(),
      body_quat_max_abs: z.number(), geom_pos_max_abs_m: z.number(),
      geom_rot_max_abs: z.number() }),
    rounding_decimals: z.number().int(),
  }).passthrough(),
  exported: z.object({ platform: z.string(), git: z.object({ sha: z.string(),
    dirty: z.boolean() }).passthrough() }),
}).superRefine((c, ctx) => {
  const n = c.frames.n;
  const nd = c.frames.dynamic_index.length;
  const bad = (m: string) => ctx.addIssue({ code: "custom", message: m });
  if (c.frames.time_s.length !== n || c.frames.success.length !== n) bad("frame arrays length");
  if (c.frames.pos.length !== n || c.frames.quat_wxyz.length !== n) bad("pose arrays length");
  if (c.frames.pos.some((f) => f.length !== nd)) bad("dynamic body count");
  if (c.frames.static_pos.length !== c.bodies.length) bad("static pose count");
  for (let k = 1; k < n; k++) if (!(c.frames.time_s[k] > c.frames.time_s[k - 1])) {
    bad(`time_s not strictly increasing at frame ${k}`);
    break;
  }
  if (c.actions.executed.length !== n - 1) bad("one executed action per transition");
  if (c.joints.length !== c.frames.joint_qpos[0]?.length) bad("joint naming");
  if (!c.bodies.some((b) => b.name === c.eef_body)) bad("end-effector body missing");
  if (c.media.video.n_frames !== n) bad("video frame count");
  for (const g of c.geoms) if (g.kind === "mesh" && (!g.mesh || !c.meshes[g.mesh])) {
    bad(`mesh of geom ${g.name} missing`);
  }
});
export type Clip = z.infer<typeof Clip>;

const path3 = z.array(vec3);
export const Decision = z.object({
  decision: z.number().int(),
  frame: z.number().int(),
  first_action_index: z.number().int(),
  n_executed: z.number().int(),
  timesteps: z.array(z.number().int()),
  prev_timesteps: z.array(z.number().int()),
  sample_in_normalized: z.array(z.array(z.array(z.number()))),
  x0_normalized: z.array(z.array(z.array(z.number()))),
  final_normalized: z.array(z.array(z.number())),
  final_env_action: z.array(z.array(z.number())),
  obs_window_normalized: z.array(z.array(z.number())),
  obs_valid: z.array(z.boolean()),
  diversity: z.object({ seeds: z.array(z.number().int()),
    chunks_normalized: z.array(z.array(z.array(z.number()))),
    env_action: z.array(z.array(z.array(z.number()))), mean_std_normalized: z.number() }),
  paths: z.object({ recorded_executed: path3, final: path3,
    final_ended: z.object({ after_action: z.number().int(), success: z.boolean() })
      .passthrough().nullable(),
    sample_in: z.array(path3), x0: z.array(path3), diversity: z.array(path3) }),
  checks: z.object({ replayed_chunk_equals_recorded_raw_policy_action: z.literal(true),
    executed_branch_equals_record: z.literal(true), trace_is_contiguous: z.literal(true) })
    .passthrough(),
});
export type Decision = z.infer<typeof Decision>;

export const Denoise = z.object({
  schema: z.literal("roboforge.observatory_denoise/1"),
  clip_id: z.string(),
  task: z.enum(["lift", "can"]),
  eval_run_id: z.string(),
  state_id: z.string(),
  partition: z.literal("eval_validation"),
  checkpoint: z.object({ sha256: z.string().length(64), step: z.number().int() }).passthrough(),
  outcome: Outcome,
  horizons: z.object({ T_o: z.number().int(), T_p: z.number().int(), T_a: z.number().int() }),
  kind: z.string(),
  decisions: z.array(Decision).min(1),
  scheduler: z.object({ steps: z.array(z.object({ timestep: z.number().int(),
    alpha_prod_t: z.number() }).passthrough()) }).passthrough(),
  labels: z.record(z.string(), z.string().nullable()),
  observation_names: z.array(z.string()),
  n_branches: z.number().int(),
}).superRefine((d, ctx) => {
  for (const x of d.decisions) {
    if (x.sample_in_normalized.length !== x.timesteps.length ||
      x.paths.sample_in.length !== x.timesteps.length || x.paths.x0.length !== x.timesteps.length) {
      ctx.addIssue({ code: "custom", message: `decision ${x.decision}: step counts differ` });
    }
    if (x.final_normalized.length !== d.horizons.T_p) {
      ctx.addIssue({ code: "custom", message: `decision ${x.decision}: chunk length != T_p` });
    }
  }
});
export type Denoise = z.infer<typeof Denoise>;

export const Clouds = z.object({
  schema: z.literal("roboforge.observatory_clouds/1"),
  clip_id: z.string(),
  frames: z.array(z.number().int()),
  points: z.array(z.array(vec3)),
  source_camera: z.array(z.array(z.number().int())),
  cameras: z.array(z.string()),
  cloud_sha256: z.array(z.string()),
  spec: z.object({ n_points: z.number().int(), size: z.number().int() }).passthrough(),
  label: z.string(),
}).superRefine((c, ctx) => {
  if (c.points.length !== c.frames.length || c.points.some((p) => p.length !== c.spec.n_points)) {
    ctx.addIssue({ code: "custom", message: "cloud count or size differs from the spec" });
  }
});
export type Clouds = z.infer<typeof Clouds>;

export const RegistryClip = z.object({
  id: z.string(), kind: z.string(), roles: z.array(z.string()), task: z.enum(["lift", "can"]),
  family: z.string(), seed: z.number().int(), state_id: z.string(), partition: z.string(),
  suite: z.string().nullable(), outcome: Outcome, n_frames: z.number().int(),
  checkpoint: z.object({ sha256: z.string().nullable(), step: z.number().int().nullable() })
    .passthrough(),
  eval_run_id: z.string(), physics_sha256: z.string(), has: z.array(z.string()),
});
export type RegistryClip = z.infer<typeof RegistryClip>;

export const Registry = z.object({
  schema: z.literal("roboforge.observatory_registry/1"),
  index_sha256: z.string().length(64),
  source_dir: z.string(),
  clips: z.array(RegistryClip).min(1),
  files: z.record(z.string(), z.string().length(64)),
  export_platform: z.object({ platform_id: z.string() }).passthrough(),
  export_git: z.object({ sha: z.string(), dirty: z.boolean() }).passthrough(),
});
export type Registry = z.infer<typeof Registry>;

export const MeshTable = z.object({
  schema: z.literal("roboforge.web_meshes/1"),
  bytes: z.number().int(),
  sha256: z.string().length(64),
  meshes: z.record(z.string(), z.object({ n_vert: z.number().int(), n_face: z.number().int(),
    lo: vec3, span: vec3, pos_offset: z.number().int(), index_offset: z.number().int(),
    index_type: z.enum(["uint16", "uint32"]), max_quantization_error_m: z.number() })),
  host_vs_container_max_abs_m: z.number(),
});
export type MeshTable = z.infer<typeof MeshTable>;

const Interval = z.tuple([z.number(), z.number()]);
export const FamilyEstimate = z.object({ estimate: z.number(), ci95: Interval,
  per_seed: z.array(z.number()), n_seeds: z.number().int(), n_states: z.number().int() });
export const Difference = z.object({ a: z.string(), b: z.string(), difference: z.number(),
  ci95: Interval, separation: z.string(), hypothesis: z.string().optional(),
  kind: z.string().optional() });
export const Bootstrap = z.object({ method: z.string().optional(), B: z.number().int(),
  families: z.record(z.string(), FamilyEstimate), differences: z.array(Difference) })
  .passthrough();
const Strip = z.object({ outcomes: z.array(z.object({ state_id: z.string(),
  category: z.string(), steps: z.number().nullable() })) }).passthrough();
export const Findings = z.object({
  schema: z.literal("roboforge.observatory_findings/1"),
  sources: z.array(z.object({ path: z.string(), sha256: z.string().length(64) })),
  notes: z.record(z.string(), z.string()),
  level1: z.object({ tasks: z.record(z.string(), z.object({
    n_states: z.number().int(), success: Bootstrap, censored_episode_length: Bootstrap,
    per_checkpoint: z.array(z.object({ family: z.string(), seed: z.number().int().nullable(),
      step: z.number().int().nullable(), successes: z.number().int(),
      n_episodes: z.number().int(), wilson95: Interval }).passthrough()),
    taxonomy_by_family: z.record(z.string(), z.record(z.string(), z.number().int())),
    cost_latency: z.record(z.string(), z.object({ latency_ms_median: z.number().nullable(),
      param_count: z.number().int().nullable() }).passthrough()),
    strips: z.array(Strip.extend({ family: z.string(), seed: z.number().int().nullable() })),
  }).passthrough()) }).passthrough(),
  rgb: z.object({ success: Bootstrap, per_checkpoint: z.array(z.object({ family: z.string(),
    seed: z.number().int(), successes: z.number().int(), n: z.number().int(),
    latency_ms_median_of_episode_medians: z.number() }).passthrough()),
    taxonomy_by_family: z.record(z.string(), z.record(z.string(), z.number().int())),
    strips: z.array(Strip.extend({ family: z.string(), seed: z.number().int() })) })
    .passthrough(),
  level2: z.object({
    placement: z.object({ bootstrap: Bootstrap.extend({ contrasts: z.array(z.object({
      name: z.string(), estimate: z.number(), ci95: Interval, separation: z.string() })
      .passthrough()) }), evaluations: z.array(z.record(z.string(), z.unknown())) })
      .passthrough(),
    heldout: z.object({ success: Bootstrap, censored_episode_length: Bootstrap })
      .passthrough(),
    taxonomy_by_arm_and_set: z.record(z.string(), z.record(z.string(), z.number().int())),
    training_generated: z.record(z.string(), z.object({ ms_per_step: z.number(),
      train_wall_s: z.number(), billed_usd: z.number(), segments: z.number().int() })
      .passthrough()),
    training_erratum: z.object({ run_id: z.string(), generated_ms_per_step: z.number(),
      corrected_ms_per_step: z.number(), completed_segment_wall_h: z.number(),
      total_wall_h: z.number(), source: z.string() }),
    strips: z.array(Strip.extend({ arm: z.string(), suite: z.string(), seed: z.number().int() })),
    placement_map: z.object({ training_square_half_m: z.number(), center_xy: z.tuple([z.number(),
      z.number()]), ood_radius_m: Interval, states: z.array(z.object({ suite: z.string(),
      state_id: z.string(), cube_initial_pos_m: vec3,
      successes_of_3_seeds: z.object({ pc_diffusion: z.number().int(),
        rgb_diffusion: z.number().int() }) })) }).passthrough(),
    privilege: z.string(), power: z.string(),
  }).passthrough(),
});
export type Findings = z.infer<typeof Findings>;

export const Verification = z.object({
  schema: z.literal("roboforge.viewer_verification/1"),
  threshold: z.object({ robot_iou_min: z.number(), object_iou_min: z.number(),
    centroid_px_max: z.number() }),
  frames: z.array(z.object({ clip: z.string(), frame: z.number().int(), camera: z.string(),
    robot_iou: z.number(), object_iou: z.number().nullable(), robot_centroid_px: z.number(),
    pass: z.boolean() }).passthrough()),
  pass: z.boolean(),
}).passthrough();
export type Verification = z.infer<typeof Verification>;
