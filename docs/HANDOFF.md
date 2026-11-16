<!-- parsable handoff header (real values; Session 16 STOP, X2-b done, updated 2026-11-16) -->
```yaml
project: roboforge-3d
current_session: 16
session_status: PASS          # Session 16: X2 phase X2-b, Can MH: preregistered search, 20 new runs, 460 validation evaluations, choices, X2 test configuration; $0
next_session: 17              # X2 phase X2-c (freeze, held-out once, analysis once); exact prompt in # Session 16 Handoff §6
last_verified_commit: 9df6628   # code and preregistration; suite and benchmark ran from it clean; STOP checks ran on it plus the Session 16 records and evidence (# Session 16 Handoff §4)
protocol_commits: {v1: 8b08b95, diffusion_amendment: 3aed462, main_budget_amendment: ec9626b, lift_search_space: 6dce01b, can_search_space: 96aa972, section14_preregistration: 2859d04}
freeze_commit: bf0a0fa          # docs/FINAL_TEST_FREEZE.md, sha256 6a3c6874c4bc1857efe46ccb05c0a1fa68247671fc9a96f5f54ce079aa9153e9, git blob bfdb4b77d478bdb1908b34b96e1dad70eb88147c
heldout_results_commit: 97cc48a # reports/session7/results/results.json sha256 b8b097c38765c588711615917e3cfd5be6847e63083ddebe92f0f1488672408c; RUN ONCE, never rerun
rgb_freeze_commit: d9c6217      # docs/RGB_EXPERIMENT_FREEZE.md sha256 47e7cb8ee70743dbcc6a0295a92a7462a2309870aebb3eda19060ca0e0ac385f; lock re-gate D-081 (378ea43)
rgb_heldout_results_commit: 3b23255   # reports/session8/rgb_results/; held-out run bc054bc; RUN ONCE, never rerun
level1_freeze: docs/LEVEL1_COMPLETE.md   # HARD SCIENTIFIC BOUNDARY: later RGB/point-cloud results never modify Level 1 selection, labels or findings
python: 3.11.15                       # development (macOS arm64); canonical platform uses 3.11.18
sim_engine_version: mujoco 3.2.7
robosuite_revision: 51cc01785bab80ffeed20da15e67d7dd4140e76a   # v1.5.1
robomimic_revision: d309eaecc18acf4152a830a895a6984b8ac71b05
diffusers: 0.11.1
torch: 2.15.0 (macOS) / 2.15.0+cu132 (container and Modal)
evaluation_platform: {platform_id: linux-aarch64-container-1f685638f28c, image_id: "sha256:1f685638f28c720fbec7b084b8e5c2d68073d8f637450d5f6e0a34f0d123071c", runner: infra/containers/run_sim_cpu.sh}
dataset_and_specs: "demonstrations, validation/held-out suites, uv.lock, pyproject unchanged since Session 2; Session 7 added data/eval_states/<task>/{robustness_base,ood_*} and configs/eval/{robustness,failure_taxonomy}.yaml (D-063, §14); Session 15 added X2 files only (data/manifests/robomimic_v1.5_mh.json, data/manifests/tasks/can_mh.json, data/splits/can_mh.json, data/normalization/can_mh_lowdim_v1.json, data/eval_states/can/x2_*; D-118 to D-123)"
raw_rgb_source: "data/raw/robomimic_v1.5/{lift,can}/ph/demo_v15.hdf5 downloaded (states + model XML per demo; no images); Square not downloaded"
main_suite:
  families: [bc_mlp, bc_rnn, bc_mlp_chunk, diffusion_unet]
  tasks: {lift: "Session 6 (24 runs, 480 evaluations)", can: "Session 7 (24 runs, 480 evaluations)"}
  seeds: [0, 1, 2]
  budget: "100,000 steps, batch 100, checkpoints every 5,000 (K = 20), every checkpoint on 50 validation starts"
  tuning: "4 of 4 configurations used per family on Lift and on Can; none remains"
  compute: "local CPU only, 1 torch thread, <= 8 concurrent training runs; canonical container <= 4 in parallel; $0 (Gate C, D-051)"
  can_suite: {plan: configs/experiments/level1_core_can.yaml, plan_sha256: 881b4fb7fd72a33866ad4924db3b72cbe4270a212b03a8cc958efc6ba8a89e3c, summary: reports/session7/suite/level1_core_can.json, runs: 24, evaluations: 480, failed: 0}
baseline_configs:
  bc_mlp:         {horizons: "T_o=1 T_p=1 T_a=1", model: "MLP 2x1024 ReLU", params: 1077255}
  bc_rnn:         {horizons: "L=10 T_p=1 T_a=1", model: "LSTM 2x296, sliding window", params: 1080703}
  bc_mlp_chunk:   {horizons: "T_o=2 T_p=16 T_a=8", model: "MLP 2x1024 ReLU", params: 1204336}
  diffusion_unet: {horizons: "T_o=2 T_p=16 T_a=8", model: "flatten encoder + ConditionalUnet1D (64,128)", params: 1092551, sampler: "DDIM-10", ema: "power 0.75"}
lift_test_configuration:              # FROZEN (bf0a0fa); reports/session6/lift_test_configuration.json (sha256 38ac3c29...)
  held_out_schedule_manifest_sha256: e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da   # n = 50; hash only, never loaded
  bc_mlp:   # chosen trial 4 (optim.weight_decay=0.1)
    s6_bc_mlp_wd1e1_lift_seed0_c39e4980: {seed: 0, selected_step: 15000, validation: 49/50, checkpoint_sha256: e68bc5e05c5b5f78c0e11c33401ed6feee361be44a629ff1fac394949cd7ef54}
    s6_bc_mlp_wd1e1_lift_seed1_628a1e7a: {seed: 1, selected_step: 5000, validation: 48/50, checkpoint_sha256: 5454cf8081b88319bcb91e35d8e69eb8fe0dbef9973c21454b690309dd9995ea}
    s6_bc_mlp_wd1e1_lift_seed2_67c90df6: {seed: 2, selected_step: 5000, validation: 49/50, checkpoint_sha256: 81a69c4c31d1446e5915977831338244e71dda7721697ef62230488548025d1e}
  bc_rnn:   # chosen trial 4 (optim.weight_decay=0.1)
    s6_bc_rnn_wd1e1_lift_seed0_166e9b0a: {seed: 0, selected_step: 15000, validation: 50/50, checkpoint_sha256: fb71d7c60f72ab68c0906d08837d51bbeeb06fbc253de0d5544c255d2d767b1b}
    s6_bc_rnn_wd1e1_lift_seed1_5221b93e: {seed: 1, selected_step: 10000, validation: 50/50, checkpoint_sha256: 30269f1f3b179640687e2c969e8c05095e1c73d1ddbcebbcd30ee5f627511508}
    s6_bc_rnn_wd1e1_lift_seed2_3368e1df: {seed: 2, selected_step: 5000, validation: 50/50, checkpoint_sha256: 1334302f18c9b6a82fa99e676965f5cb54896220b6c8a47b505e727b902dc9b9}
  bc_mlp_chunk:   # chosen trial 2 (optim.schedule=cosine)
    s6_bc_mlp_chunk_cos_lift_seed0_445dbbb5: {seed: 0, selected_step: 15000, validation: 50/50, checkpoint_sha256: c72c4d182450e63456ce53d1db6f35a95531ecab79181602956e6feaaeb03d54}
    s6_bc_mlp_chunk_cos_lift_seed1_c5955411: {seed: 1, selected_step: 5000, validation: 49/50, checkpoint_sha256: 9a46001eab065a990e80270f37e622a4c86664aec75517ac5acbab018dc491bb}
    s6_bc_mlp_chunk_cos_lift_seed2_98c39779: {seed: 2, selected_step: 10000, validation: 50/50, checkpoint_sha256: b7c9aa48bbd063f4466cc9f855542fa7408d181a13a7366aed6c75485ab2fe21}
  diffusion_unet:   # chosen trial 3 (optim.lr=3.0e-4)
    s6_diffusion_unet_lr3e4_lift_seed0_2f635398: {seed: 0, selected_step: 100000, validation: 50/50, checkpoint_sha256: 7e48766542774f52a5c9e73ca2a2776d95981188ed30d65188746479c55f0f38}
    s6_diffusion_unet_lr3e4_lift_seed1_50ae45de: {seed: 1, selected_step: 95000, validation: 50/50, checkpoint_sha256: f26868ea80e307200e92de962493797add56e97d0a1c38519383de64baeb81f7}
    s6_diffusion_unet_lr3e4_lift_seed2_427ac4f5: {seed: 2, selected_step: 80000, validation: 50/50, checkpoint_sha256: b52c30177478e6c8fed6eabd220c08fb839974550da925cb751b341a537f5587}
can_test_configuration:               # FROZEN (bf0a0fa); reports/session7/can_test_configuration.json (sha256 116f10a4...)
  held_out_schedule_manifest_sha256: 5f407748f5f261ef8fdfe4bec0d53c74f8e5ae064e53ff831e17a649528f7ff0   # n = 50; tested once in Session 7
  bc_mlp:   # chosen trial 2 (optim.schedule=cosine)
    s7_bc_mlp_cos_can_seed0_ae8152e3: {seed: 0, selected_step: 55000, checkpoint_sha256: b8fa2d06c19ae9d4830cf6b2a50b272e6fc91f3ab8668c68aae17092a16dc081}
    s7_bc_mlp_cos_can_seed1_00564355: {seed: 1, selected_step: 15000, checkpoint_sha256: cbf55c0d0656506c8457519024dee360d52f06abf3e92a485a2ef38ee03a34e2}
    s7_bc_mlp_cos_can_seed2_7359a97f: {seed: 2, selected_step: 55000, checkpoint_sha256: 3c1535918fb744fca9594da6e11220922db759f72fa09c984ec00006f0e4647a}
  bc_rnn:   # chosen trial 2 (optim.schedule=cosine)
    s7_bc_rnn_cos_can_seed0_5ff5ce91: {seed: 0, selected_step: 30000, checkpoint_sha256: 5085d716f285631e749a9439151c3aafab6fb6c67baae78800b3a9a0014e6137}
    s7_bc_rnn_cos_can_seed1_d34c6b99: {seed: 1, selected_step: 25000, checkpoint_sha256: a69eef409a35adbcd6e393e3397fb5603670a254d1d2b05b6d68e677d39fd23a}
    s7_bc_rnn_cos_can_seed2_8452a37a: {seed: 2, selected_step: 35000, checkpoint_sha256: 7cda91e6881c5d9a71a8c4b4fc5e56b6d527887b049af91eb481d700e164a148}
  bc_mlp_chunk:   # chosen trial 4 (optim.weight_decay=0.1)
    s7_bc_mlp_chunk_wd1e1_can_seed0_0c2f9d4b: {seed: 0, selected_step: 95000, checkpoint_sha256: 1968ad1148e4f20f82fffd6dfa57196f82ee3c7544e4e8c6eea5af2d7b367cd9}
    s7_bc_mlp_chunk_wd1e1_can_seed1_92a1f312: {seed: 1, selected_step: 85000, checkpoint_sha256: 3b306ad736f3b2df7e53928599b17125f834c1260224dbde97d57cdb23dc4fb2}
    s7_bc_mlp_chunk_wd1e1_can_seed2_63847bb4: {seed: 2, selected_step: 30000, checkpoint_sha256: ebbff8e38a713c584cbef4058534d0377d108701457d79ac25814bc7ff24552f}
  diffusion_unet:   # chosen trial 2 (optim.schedule=cosine)
    s7_diffusion_unet_cos_can_seed0_99251da3: {seed: 0, selected_step: 65000, checkpoint_sha256: 41dd66ddaf7b54d7114cbfda8058ede8270abbd83eefe715d679aa8645c35c1d}
    s7_diffusion_unet_cos_can_seed1_51f8481d: {seed: 1, selected_step: 100000, checkpoint_sha256: a687d9fbadf65410cc768ee63b901c248099d9de1b13bf65384433df58f74fab}
    s7_diffusion_unet_cos_can_seed2_9b0fde56: {seed: 2, selected_step: 90000, checkpoint_sha256: 49d73c93ed0a70b79a433892808468005004f89f8c923685c4cb342e3a456e27}
x2_test_configuration:                # X2-b (Session 16); reports/session16/x2/suite/x2_test_configuration.json (sha256 387ecbe5...); X2-c's X2_FREEZE.md copies it
  heldout_suite: "can/x2_heldout, 100 starts, content 40b8228c3e08cf6e3bdcf23c77a28cb273e5bd7f2eb7a797b0e7d7caebdd833c; never stepped yet"
  bc_mlp:   # chosen trial 1 (base configuration)
    s16x2_bc_mlp_can_seed0_ad021ce7: {seed: 0, selected_step: 85000, validation: 16/50, checkpoint_sha256: 83cbe5682fc8c230bb12425f419a8cfa1575cb20b3f26da0848942d4a0feec25}
    s16x2_bc_mlp_can_seed1_8bf126a6: {seed: 1, selected_step: 45000, validation: 12/50, checkpoint_sha256: e16cfb3f63916355b26914e66f13fe2662d4721652ff87f8f44679dcd84ea0fe}
    s16x2_bc_mlp_can_seed2_8324158e: {seed: 2, selected_step: 75000, validation: 15/50, checkpoint_sha256: 61571865ff9e1e875ebd6b16108f6a41c76701d819c3bb6b13323cb2102365bc}
  bc_rnn:   # chosen trial 4 (optim.weight_decay=0.1)
    s16x2_bc_rnn_wd1e1_can_seed0_14e39a6c: {seed: 0, selected_step: 65000, validation: 36/50, checkpoint_sha256: fb6d3ccdab444dfd0d87bb11db60ea8b963efd50ad9960e0fc3045d2b1474c7a}
    s16x2_bc_rnn_wd1e1_can_seed1_7eddb00a: {seed: 1, selected_step: 65000, validation: 36/50, checkpoint_sha256: 9a856115af4f9370b444993332c942585cf86923e751c36954d791382b9d2d41}
    s16x2_bc_rnn_wd1e1_can_seed2_14ae1f29: {seed: 2, selected_step: 90000, validation: 39/50, checkpoint_sha256: 77ac0d5d9b32fcadca7321b6781d61f503a333e444231716b00465087fd9db5f}
  bc_mlp_chunk:   # chosen trial 2 (optim.schedule=cosine)
    s16x2_bc_mlp_chunk_cos_can_seed0_7d7745e8: {seed: 0, selected_step: 15000, validation: 49/50, checkpoint_sha256: a47c676a00a9051e1b4825b6ff07e10a9f0508a6dce3ebbe38a00fd52e3b0078}
    s16x2_bc_mlp_chunk_cos_can_seed1_96b63a9d: {seed: 1, selected_step: 5000, validation: 50/50, checkpoint_sha256: b9e0b945be90607f8314795de753f54ddcd4bfc4b5fb0af16ab9ad7c6214bb8b}
    s16x2_bc_mlp_chunk_cos_can_seed2_a6007970: {seed: 2, selected_step: 30000, validation: 48/50, checkpoint_sha256: 218afa96f846ef55a03b6e763f48f6aae83b735c7d62e918f2b7d4c06b29ca0d}
  diffusion_unet:   # chosen trial 2 (optim.schedule=cosine)
    s15x2_diffusion_unet_cos_can_seed0_99251da3: {seed: 0, selected_step: 100000, validation: 48/50, checkpoint_sha256: 1301c3b26eedf0ae72c550b13549caf09aa739d74ba580e2314f7fd28838f357}
    s16x2_diffusion_unet_cos_can_seed1_51f8481d: {seed: 1, selected_step: 90000, validation: 46/50, checkpoint_sha256: c39fe169012fd060c47080ef30ca08fcb25ba1010ee189765d498fb33053f7e2}
    s16x2_diffusion_unet_cos_can_seed2_9b0fde56: {seed: 2, selected_step: 65000, validation: 47/50, checkpoint_sha256: d04e02f7de5538a08ca51bc83b42dd4e1df51c320955b6d03d613e8bfbc2830a}
ablation_a3_runs: [s7_diffusion_unet_cos_to1_can_seed0_4b9d434e, s7_diffusion_unet_cos_to1_can_seed1_f9e76b7e, s7_diffusion_unet_cos_to1_can_seed2_5fa43638]   # T_o = 1; selected 95k/100k/90k, 49/50 each (validation, development)
heldout_results:                      # family success %, 3 seeds x 50 held-out starts (reports/session7/results/tables.md)
  lift: {bc_mlp: 90.7, bc_rnn: 100.0, bc_mlp_chunk: 98.0, diffusion_unet: 97.3, random_uniform: 0.0}
  can:  {bc_mlp: 76.7, bc_rnn: 96.0, bc_mlp_chunk: 96.0, diffusion_unet: 95.3, random_uniform: 0.0}
modal: {gate_a: done, A-01: "closed, $0.148", A-02: "closed, $0.0317", A-03: "closed, $8.06", A-04: "Level 2, closed, $2.37 (5 launches; KI-047; detached launches since D-085)", open_approvals: none, month_to_date: "$15.89 of $30 (2026-11-09)", egl_vs_osmesa: "PENDING (KI-001, KI-033)"}
final_test: "Level 1 TESTED ONCE (Session 7); RGB study TESTED ONCE (Session 8, D-081); Level 2 TESTED ONCE (Session 9, D-086 gate, a031e88; analysis 620c541). Session 10 re-rendered recorded episodes only. All three locks are open because their preconditions are committed; that is not permission: no further held-out rollout of any Level 1, RGB or Level 2 configuration, ever. Session 13 (X1) stepped only the Session 7 robustness sets. Session 14 (X3) stepped only the Can validation starts and the new can/x3_heldout suite (24 evaluations, once); isolation PASS. Sessions 15-16 (X2-a, X2-b) stepped only can/x2_validation (20 + 460 evaluations); no can/x2_heldout start yet; isolation PASS"
disk: "about 18 GiB free after the Session 16 suite (KI-008; swap files share the volume); runs/x2 takes 19.59 GiB (X2-b added 17.96); X2-c needs well under 1 GiB (12 evaluations of 100 episodes, estimate); X3 runs, validation directories and runs/x3 take 7.1 GiB; the Observatory export is about 19 MB in git; the Modal runs Volume keeps every checkpoint"
next_required_owner_approval: "none for X2-c, which is fully preregistered (docs/extensions/X2_PREREGISTRATION.md §7); whether X1, X2 and X3 are published (v1.1) is the owner's decision (D-105). No paid compute is open; any Modal job needs a new approval row"
observatory: {decisions: "D-087-D-091", site: website/, export: "reports/session10/observatory/ (37 clips from clean b5fc2ee, canonical container; ebfd0c3)", findings: "findings.json, 8 frozen sources (c73a080)", verification: "viewer_verification.json: 18/18 frames pass (robot IoU >= 0.9998, centroid <= 0.04 px)", hosting: "none (owner, D-090); no deployed or staging URL", meshes: "extracted at build time by make web-assets, gitignored (D-090)", provenance: docs/VISUAL_PROVENANCE.md}
level2_study: {freeze: "docs/LEVEL2_EXPERIMENT_FREEZE.md @ b93c50c (sha256 ffe5a03e...)", gate: "D-086 @ b1f4da2, amended 08c3aa8", runs: [s9_pc_diffusion_dp3_lift_seed0_8fd35613, s9_pc_diffusion_dp3_lift_seed1_a66a2758, s9_pc_diffusion_dp3_lift_seed2_402a1f24], selections: "30000 / 40000 / 60000 (reports/session9/pc_selection/)", placement: "reports/session9/level2_placement/ @ 4ebebc1", heldout: "reports/session9/pc_heldout/ @ a031e88", results: "reports/session9/level2_results/ @ 620c541 (+ ERRATA.md)", h5: "pc - rgb on ood_cube_shift -41.3 pp [-54.0, -28.0], separated (-)", heldout_success: {pc_diffusion: 100.0, rgb_diffusion: 100.0}, label: DP3-inspired, privilege: sensor-like}
paper: {pdf: paper/main.pdf, sha256: c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f, pages: 23, build: "make paper (Tectonic 0.17.0, --only-cached, SOURCE_DATE_EPOCH=1794182400)", generated: "paper/generated/ (35 files, index.json)", audit: docs/AUDIT.md, decisions: "D-092-D-096", label: "local build, not peer reviewed", fresh_clone: "byte-identical rebuild (Session 12)"}
release: {version: v1.0.0, audit: docs/FINAL_RELEASE_AUDIT.md, decisions: "D-097-D-104", repository: "https://github.com/zackholt05-tech/roboforge", observatory: "GitHub Pages (D-102)", ci: ".github/workflows (D-101)", state: "see # Session 12 Handoff §5 (publication record, verified)"}
rgb_study: {freeze: "docs/RGB_EXPERIMENT_FREEZE.md @ d9c6217", gate: "D-081 @ 378ea43", selections: reports/session8/rgb_selection/, heldout: "reports/session8/rgb_heldout/ @ bc054bc", results: "reports/session8/rgb_results/ @ 3b23255", success: {rgb_diffusion: 100.0, rgb_bc_chunk: 99.3}, diffusion_minus_bc: "+0.7 pp [+0.0, +4.0], not separated", videos: reports/session8/rgb_videos/}
x1_study: {decision: D-107, label: "exploratory, after the freeze", code: 09d1a90, preregistration: "docs/extensions/X1_PREREGISTRATION.md @ ab02469", evaluations: "45 once from ab02469 (reports/session13/x1_clamp/evals/, runs/x1_clamp/)", analysis: "reports/session13/x1_clamp/ @ c2a9ae6", primary: "Lift ood_arm_offset drop(diffusion) - drop(clamped BC-MLP-chunk) -40.0 pp [-54.0, -25.3], separated (-) (frozen -40.0 [-54.0, -26.0]); reading R1", clamped_minus_unclamped: "0 of 15 separated; largest |diff| 0.7 pp", engineering_checks: "logs/session13/x1_checks/ (all PASS)"}
x3_study: {decisions: "D-109-D-115", label: "separate replication", preregistration: "docs/extensions/X3_PREREGISTRATION.md @ fb9f05e", heldout_suite: "can/x3_heldout, 100 starts, content 9c199e19 (81afa3b)", training: "12 runs s14_* (4 families x seeds 3-5), 0 failed", validation: "240 evaluations, 12 selections (reports/session14/suite/x3_validation.json @ 86e6c9b; KI-062, D-112)", evaluations: "24 once from 13c5db2 (reports/session14/x3/evals/, runs/x3/), isolation PASS", analysis: "reports/session14/x3/ @ 335c27c", primary: "H2 diffusion - BC-MLP +22.7 pp, H3 BC-RNN - BC-MLP +23.8 pp: separated under bootstrap and seed-level (holds, robust); H2 diffusion - BC-RNN -1.2, H2-c -1.7: not separated (unchanged)"}
x2_study: {decisions: "D-116-D-128", preregistration: "docs/extensions/X2_PREREGISTRATION.md @ 9df6628 (sha256 d9bc5955...)", option: "A, Can MH", phase: "X2-a done (Session 15); X2-b done (Session 16); X2-c next", x2b: "reports/session16/x2/suite/ from clean 9df6628: chosen diffusion t2 (48), BC-RNN t4 (36), BC-MLP-chunk t2 (49), BC-MLP t1 (16)", data: "data/manifests/robomimic_v1.5_mh.json (hash-verified); split data/splits/can_mh.json (operator-stratified, within-operator claims only)", suites: "can/x2_validation 50 starts 47b889cd, can/x2_heldout 100 starts 40b8228c (4459c00); no held-out start stepped", pilot: "reports/session15/x2/pilot/ from fe9df74; best of 5 of 50: diffusion 48, BC-MLP-chunk 48, BC-RNN 34, BC-MLP 6; reading headroom (D-124)", label: "development numbers on X2 validation starts; not a result"}
```

# Next: Session 17, X2 phase X2-c (freeze, held-out once, analysis once)

Session 16 ran X2 phase X2-b (# Session 16 Handoff below). Read `docs/extensions/README.md`,
`docs/extensions/X2_harder_task.md` and `docs/extensions/X2_PREREGISTRATION.md`. The exact
Session 17 prompt is in # Session 16 Handoff §6. Every earlier boundary still applies, and the
X2-b evidence is now fixed too:
- never rerun `make x2b-suite` or `make x2b-benchmark`;
- never edit `reports/session16` results;
- never prune or move `runs/x2`.

| Study | File | Session |
|---|---|---|
| X1 evaluation-only action-clamp test (KI-052) | `docs/extensions/X1_action_clamp.md` | 13, DONE |
| X2 a harder or multimodal task: option A, Can MH (D-116) | `docs/extensions/X2_harder_task.md` | 15 X2-a DONE; 16 X2-b DONE; X2-c next |
| X3 more training seeds, on a fresh held-out suite | `docs/extensions/X3_more_seeds.md` | 14, DONE |

# Session 16 Handoff (X2-b)

Do not trust this file blindly. Every claim names its command and evidence
(`logs/session16/`, `reports/session16/`, LEDGER E16-01 to E16-06).

## 1. Session completed
Session 16: post-release extension X2, phase X2-b (tuning, training and selection on validation
starts), option A, Can MH (`docs/extensions/X2_harder_task.md` §4; D-105, D-115 to D-128).
Status **PASS**. Every X2 number so far is a development number on the 50 `x2_validation`
starts. No X2 held-out start was stepped.

## 2. Completed work
- **Owner decisions.** D-125: proceed on Can MH; the four pilot runs count as their trials'
  seed-0 runs; the host limits; a disk reclaim. D-128: two scratch runs and the download
  caches reclaimed.
- **Fixed decisions.** D-126 (Makefile `PREFIX ?= s16`, evidence session 16, `x2b-*`
  targets); D-127 (the design; outcome note at the STOP).
- **Preregistration.** `docs/extensions/X2_PREREGISTRATION.md` (`d9bc5955…`), committed in
  `9df6628` before the first X2-b training. It fixes the X2-c analysis (§7).
- **Code** (new files only, in `9df6628`; `make freeze-check` unchanged):
  - `src/roboforge/x2/{suite,host}.py`;
  - `scripts/x2b_{train,eval,suite,report,benchmark}.py`;
  - `configs/experiments/x2_can_mh.yaml`, `configs/extensions/x2b.yaml`;
  - tests.
- **Suite** (E16-04). `make x2b-suite` ran once from clean `9df6628`, 24.1 h: 20 new runs
  and 460 new evaluations, with none failed.
- **Benchmark, verification, isolation** (E16-05). Latency on the canonical container. All
  480 summaries and the 12 selected checkpoints verified
  (`logs/session16/x2/suite/verification.txt`). Isolation PASS.
- **Records.** KI-008 and KI-035 updated; LEDGER E16-03 to E16-06; STATUS;
  `docs/extensions/README.md`.

## 3. Results (development numbers on the 50 `x2_validation` starts; never a result)
`reports/session16/x2/suite/tables.md` has every run. Configurations were chosen by the frozen
§6.3 rule over the four seed-0 trials. Checkpoints were selected by the frozen §7 rule per run.

| Family | Trial 1 / 2 / 3 / 4 (seed 0, selected successes) | Chosen | Seed 0 / 1 / 2: step (successes) |
|---|---|---|---|
| diffusion | 48 / **48** / 47 / 45 | 2, cosine (tie; mean first-success step 197 vs 206) | 100k (48) / 90k (46) / 65k (47) |
| BC-RNN | 34 / 34 / 35 / **36** | 4, weight decay 0.1 | 65k (36) / 65k (36) / 90k (39) |
| BC-MLP-chunk | 49 / **49** / 48 / 48 | 2, cosine (tie; 156 vs 172) | 15k (49) / 5k (50) / 30k (48) |
| BC-MLP | **16** / 11 / 14 / 9 | 1, base | 85k (16) / 45k (12) / 75k (15) |

- **Ceiling.** Diffusion and BC-MLP-chunk sit at 46–50 of 50 on validation, so H2c may not
  separate on the held-out starts (preregistration §8). The single-step families leave
  headroom.
- **Selection optimism.** Each number is the maximum over 20 checkpoints on the same 50
  starts. BC-MLP-chunk seed 1 was selected at 5k steps (50/50). The 100 held-out starts are
  unbiased for the selected checkpoints.
- **Cost, measured** (estimates from the preregistration §9 in brackets):
  - training 33.86 training-hours summed [30.5];
  - evaluations 68.98 container-hours, mean 540 s [about 64];
  - wall 24.1 h [about 23];
  - `runs/x2` +17.96 GiB [18–23];
  - $0.
- **Latency** (observe + act median per decision; `reports/session16/x2/suite/latency/`):
  BC-MLP 1.4–1.6 ms, BC-MLP-chunk 1.7–1.8 ms, BC-RNN 9.4–9.8 ms, diffusion 48–53 ms.
- **Host.**
  - The launch guard held 10 times. The swap rate still peaked at 29,285 pages/s with only
    evaluation containers running (KI-035).
  - The replicate disk gate held twice and cleared without intervention. Free disk was
    11.0 GiB at its lowest (KI-008).

## 4. STOP checks
On `9df6628` plus the Session 16 records and evidence (documents, logs and reports only; no code changed), host, no container running: `make lint` clean; full suite alone **644 passed** (9:24; equal to the Session 16 baseline; `logs/session16/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged"); `tests/research_integrity` 69 passed; `make release-check` PASS with the new evidence staged so that its scan covers it (139 disclosed local-path files, D-100 and D-114; no new local path). Evidence verification and isolation PASS (E16-05). No frozen file, freeze, result or executed outcome changed; no held-out start of any suite was stepped; nothing was pushed, tagged, released or deployed.

## 5. Owner decisions pending (none blocks X2-c)
1. Whether to publish X1, X3 and, after X2-c, X2 (v1.1; D-105). Nothing was pushed, tagged
   or deployed.
2. The optional diffusion side of KI-052 (still open from Session 14).

## 6. Exact prompt for Session 17 (X2, phase X2-c)

```text
Implement RoboForge 3D SESSION 17 ONLY: post-release extension X2, phase X2-c (freeze, the X2
held-out suite once, the preregistered analysis once; option A, Can MH), as specified in
docs/extensions/X2_harder_task.md §4 and docs/extensions/X2_PREREGISTRATION.md §7, under
DECISIONS D-105 and D-115 to D-128 and the rules of MASTER_SPEC.md.

Begin by reading all of MASTER_SPEC.md, docs/extensions/README.md,
docs/extensions/X2_harder_task.md, docs/extensions/X2_PREREGISTRATION.md (all of it,
especially §4, §7 and §8), docs/FINAL_TEST_FREEZE.md (the template for X2_FREEZE.md; §4 the
engineering-incompleteness rule), docs/STATUS.md, docs/HANDOFF.md (# Session 16 Handoff),
docs/DECISIONS.md (especially D-095, D-105, D-111 to D-128 and D-127's outcome note),
docs/KNOWN_ISSUES.md (KI-008, KI-028, KI-035, KI-043, KI-053, KI-061 to KI-064),
docs/EXPERIMENT_LEDGER.md (Sessions 14 to 16), reports/session16/x2/suite/tables.md and
reports/session16/x2/suite/x2_test_configuration.json, and inspect the actual repository and
test state. Do not blindly trust handoff claims; confirm every prerequisite with executable
checks:
- x2_test_configuration.json has SHA-256 387ecbe5ab77ade9dd39ae47917194780822928d0f48ee660ed2122a9fb1b863;
- its 12 checkpoint files re-hash to it and to their run manifests;
- make check-x2-data and make check-x2-states pass (can/x2_heldout content 40b8228c…);
- no summary for any can/x2_heldout start exists anywhere;
- runs/x2 is intact (20 s16x2 runs, 4 s15x2 runs, 480 evaluation records);
- make freeze-check is unchanged.
Report free disk, swap use and running containers. X2-c needs no owner decision unless a
prerequisite fails; if one fails, stop and ask the owner.

1. Record decisions before acting: Makefile defaults PREFIX ?= s17 and evidence session 17,
   and the X2-c targets with their code.
2. New code in new files only (KI-064):
   - an X2 held-out evaluator. The X2-b evaluator refuses x2_heldout by design
     (check_x2_eval_suite), so this evaluator reuses the X2 evaluation core but admits only
     can/x2_heldout and only the 12 checkpoints listed in X2_FREEZE.md. It verifies each
     checkpoint's SHA-256 before loading it. It refuses to run if the tree is dirty, if
     X2_FREEZE.md is not committed, or if any held-out summary already exists.
   - an analysis script for the preregistration §7 exactly:
     - the outcome matrices (3 seeds × 100 starts per family);
     - the frozen hierarchical paired bootstrap, unchanged: B = 10,000, seed 20261106, the
       frozen draw order;
     - the D-095 Welch seed-level interval;
     - the four contrasts (H2 diffusion − BC-MLP, H2 diffusion − BC-RNN, H2c diffusion −
       BC-MLP-chunk, H3 BC-RNN − BC-MLP), with the verdict rules;
     - the side-by-side relation to the frozen Level 1 Can PH verdicts;
     - the secondary outputs.
   - tests on synthetic outcomes first (the KI-062 lesson), and a research-integrity test.
     The test checks that only the X2-c code path can reach x2_heldout, and only through
     X2_FREEZE.md.
   Run make freeze-check before and after every code change, and tests/research_integrity on
   the new code.
3. Write and commit docs/extensions/X2_FREEZE.md, with docs/FINAL_TEST_FREEZE.md as the
   template, before any held-out rollout. It contains:
   - a verbatim copy of the X2 test configuration (12 checkpoints with SHA-256 and validation
     successes);
   - the held-out suite hash;
   - the evaluator and analysis code hashes;
   - the commit order.
   Commit it from a clean tree.
4. Run the 12 evaluations once on the canonical container, from the clean commit that contains
   X2_FREEZE.md. That is all 100 can/x2_heldout starts, policy seed 0, R = 1, 500-step
   episodes. Record isolation before and after. There is no automatic rerun: an evaluation
   that fails before writing its summary is rerun once, only for a fault outside the policy,
   as a recorded decision.
5. Run the analysis once and write reports/session17/x2/ (results, tables, analysis record).
   Label it "X2 (Can MH)": reported beside the frozen Level 1 Can PH result, never instead of
   it, never pooled, within-operator only.

Binding boundaries:
- Frozen: Level 1, the RGB study, Level 2, X1, X3, and the X2-a and X2-b evidence.
- Never rerun: make eval-final, make analyze-final, make rgb-heldout,
  scripts/analyze_rgb_heldout.py, make pc-heldout, make pc-placement, make level2-analyze,
  scripts/analyze_robustness.py, make x1-run, make x1-analyze, make x3-states, make x3-train,
  make x3-validate, make x3-check, make x3-run, make x3-analyze, make x2-download,
  make x2-states, make x2-pilot, make x2b-suite or make x2b-benchmark.
- Never rebuild the X2 data with --update-committed.
- Never edit:
  - reports/session6-10, reports/session13, reports/session14, reports/session15 or
    reports/session16 results;
  - the freezes, the X1, X2 or X3 preregistration, LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's
    executed outcome;
  - X2_FREEZE.md after its commit;
  - any file that make freeze-check pins.
- Never prune or move frozen runs: runs/x1_clamp, runs/x1_checks, runs/x3, runs/x3_checks,
  runs/x2 and the s14_* runs included.
- After the 12 held-out evaluations: no further X2 held-out rollout, ever.
- No paid or remote compute, push, tag, release or deployment without explicit owner
  authorization.
- No changes to existing data, eval suites, uv.lock or Python dependencies without a decision.
- Never uv sync or uv run inside the image.
- Use EVIDENCE_SESSION=17. Every container job runs via run_sim_cpu.sh with
  ROBOFORGE_EVIDENCE_SESSION=17, from a clean tree.
- Never run the full test suite alongside a container job (KI-043).
- Label every estimate as an estimate, in hours from measured times: the 12 evaluations of
  100 starts take about 3.6 container-hours, from the X2-b per-family means per 50 starts
  (diffusion 642 s, BC-RNN 667 s, BC-MLP-chunk 228 s, BC-MLP 631 s). That is above the
  preregistration §9's 2–3 container-hours.

At the SESSION 17 STOP boundary:
- update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and EXPERIMENT_LEDGER;
- run make lint, the full suite alone, make freeze-check, tests/research_integrity and
  make release-check;
- commit verified work as Zackary Holt <zackholt05@gmail.com> with no AI co-author trailers;
- inspect git log --format=fuller -n 5;
- ask the owner whether to publish X1, X2 and X3 (v1.1, D-105), and write the exact prompt
  for the session the owner chooses;
- STOP. Do not proceed automatically.
```

# Session 15 Handoff (X2-a)

Do not trust this file blindly. Every claim names its command and evidence
(`logs/session15/`, `reports/session15/`, LEDGER E15-01 to E15-07).

## 1. Session completed
Session 15: post-release extension X2, phase X2-a (data and contract), option A, Can MH
(`docs/extensions/X2_harder_task.md`, D-105, D-115, D-116). Status **PASS**. Every X2 number
so far is a development number on X2 validation starts; no X2 held-out start was stepped.

## 2. Completed work
- **Owner decision.** D-116 (option A, Can MH).
- **Fixed decisions.** D-117 (Makefile `PREFIX ?= s15`, evidence session 15, X2 targets);
  D-118 (new manifest `data/manifests/robomimic_v1.5_mh.json`, SHA-256 from the pinned
  revision, verified on download); D-119 (split by robomimic's operator-stratified masks);
  D-120 (horizon 500); D-121 (X2 data artifacts and the in-process binding of frozen code);
  D-122 (`demo_81` kept as an accepted flag); D-123 (the two X2 suites); D-124 (the pilot;
  outcome note added).
- **Data** (E15-02, E15-03): both MH files downloaded and hash-verified. 300 demos, 62,756
  samples, lengths 98–1,050 (median 178). Six operator masks of 50 demos each
  (`{better,okay,worse}_operator_{1,2}`), each split 45/5 by robomimic; nothing else
  identifies an operator, so every X2 claim is within-operator (MASTER_SPEC §3.3).
  Normalization on the 270 training demos only. `docs/DATA_PROVENANCE.md` has the section.
- **Contract** (E15-03): the low-dim file's `env_args` equal Can PH's; the observation and
  action specs hash equal; open-loop replay 285/300; `RoboForgeEnv('can')` carries the X2
  task spec and horizon 500 only inside the binding
  (`reports/session15/x2/contract.json`).
- **Suites** (E15-04): `can/x2_validation` (50 starts, seeds 15,000,000 + i, `47b889cd…`) and
  `can/x2_heldout` (100 starts, seeds 16,000,000 + i, `40b8228c…`), generated on the
  canonical container, checked bitwise, committed before any rollout (`4459c00`), re-checked
  from a clean tree.
- **Headroom pilot** (E15-05, E15-06; D-124): once, from clean `fe9df74`; 4 trainings and 20
  evaluations on `x2_validation`, every exit 0; isolation PASS.
- **Code** (new files only; `make freeze-check` unchanged before and after every change):
  `src/roboforge/x2/{data,binding,contract,inspect,states,pilot,evaluate}.py`,
  `scripts/x2_*.py`, `scripts/generate_x2_states.py`, `configs/extensions/x2_pilot.yaml`, tests (E15-03 to E15-05 list SHA-256).
- **Records.** KI-008 and KI-035 updated; KI-063 (`demo_81`) and KI-064 (the frozen CLIs
  cannot see the X2 binding) new; LEDGER E15-01 to E15-07; STATUS;
  `docs/extensions/README.md`.

## 3. Results (development numbers on the 50 `x2_validation` starts; never a result)
| Family | 20k | 40k | 60k | 80k | 100k | Best of 5 | Training wall h (steps/s) | Peak RSS GiB |
|---|---|---|---|---|---|---|---|---|
| diffusion | 25 | 38 | 42 | 46 | 48 | 48 | 4.62 (6.0) | 0.37 |
| BC-MLP-chunk | 47 | 37 | 46 | 48 | 39 | 48 | 0.19 (146) | 0.39 |
| BC-RNN | 0 | 5 | 29 | 31 | 34 | 34 | 1.11 (25) | 1.08 |
| BC-MLP | 0 | 2 | 5 | 5 | 6 | 6 | 0.17 (165) | 0.33 |

- **Reading (D-124 rule): headroom.** Two families reach 48 and the "saturated" label needs
  three, so the margin is one family. The top two are at the 50-start ceiling on best-of-5
  validation: X2-b may separate the single-step families (BC-MLP, BC-RNN) from the others but
  may not separate diffusion from BC-MLP-chunk.
- **Single-step stall (observation, cause open).** At 20k, BC-MLP and BC-RNN run every episode
  to 500 steps: actions shrink after the first reach and the gripper rarely closes. Training
  losses are normal. MH actions are about half as large as PH's (median |a_xyz| 0.146 vs
  0.271), and 0.4 % of them are near zero.
- **BC-MLP-chunk varies 37–48** across checkpoints 20k apart, so a §7 choice on 50 starts is
  noisy.
- **Cost, measured:** evaluations 2.8 container-hours (mean 503 s per 50 starts, 205–819 s);
  disk 1.63 GiB; $0.
- **Host:** the swap rate peaked at 4,874 pages/s, 3 minutes after launch, with 4 trainings
  and 2 evaluation containers running. That is above the pilot's 3,000 guard, which is checked
  only at launch, so no hold fired (KI-035).

## 4. STOP checks
On `fe9df74` plus the Session 15 records (documents and evidence only), host, no container running: `make lint` clean; full suite alone **631 passed** (9:03; 28 more than Session 14's 603; `logs/session15/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged", also before and after every code change); `tests/research_integrity` 64 passed; `make release-check` PASS (139 disclosed local-path files, D-100 and D-114; no new local path). No frozen file, freeze, result or executed outcome changed; no held-out start of any suite was stepped; nothing was pushed, tagged, released or deployed.

## 5. Owner decisions pending (before any X2-b training)
1. **Go or stop.** The reading is headroom, by one family. Options: proceed with X2-b on Can
   MH as below; or go back to D-116's alternatives (Square PH or MH, new infrastructure).
2. **The pilot runs.** D-124 says X2-b may count them as "tuning trial 1". In the Level 1
   search space (D-055, D-061) they are the seed-0 runs of trial 2 (cosine: diffusion, BC-RNN,
   BC-MLP) and trial 4 (weight decay 0.1: BC-MLP-chunk), not trial 1 (D-124 outcome note).
   Choose: count each as the seed-0 run of the trial whose configuration it has (its other 15
   checkpoints then evaluated like any trial's; saves 6.1 training-hours and 4.6 h of wall
   time), or retrain all 4 trials per family under a new prefix.
3. **Concurrency** (KI-035). The pilot's concurrency (4 trainings, 2 evaluation containers
   while diffusion trains, 4 after, a third-party container) exceeded the swap guard once.
   Choose one or more: a guard that samples while jobs run and holds launches; at most 1
   evaluation container beside 4 trainings; no third-party containers during X2-b.
4. **Disk** (KI-008). 24 GiB free with a 4 GiB floor. X2-b needs about 17.5 GiB (estimate,
   biased low). A reclaim is a precondition: the uv cache (36 GB; `uv cache prune`) and/or the
   Docker build cache (2.4 GB reclaimable). Frozen runs are never pruned or moved.
5. Still open from Session 14: whether to publish X1 and X3 (v1.1, D-105); the optional
   diffusion side of KI-052. Nothing was pushed, tagged or deployed.

**Proposed X2-b design** (for the X2-b preregistration; the owner or the X2-b session
decides, before any X2-b training):
- **Search space:** Level 1's, unchanged (D-055, D-061). Per family: trial 1 is the base
  configuration, trial 2 the cosine schedule, trial 3 lr 3e-4 and trial 4 AdamW weight decay
  0.1. Architecture, horizons, parameter counts, batch and budget are fixed. It was chosen
  before the pilot outcome.
- **Budget:** the frozen one, equal for all families: 100,000 steps, batch 100, checkpoints
  every 5,000 (K = 20), and every checkpoint on all 50 `x2_validation` starts. Then:
  - §7 selection per run (`roboforge.eval.selection.select_checkpoint`, unchanged);
  - §6.3 choice per family (`roboforge.eval.tuning.choose_configuration`, unchanged; tuning
    seed 0, KI-062);
  - the chosen configuration then trains seeds 1 and 2.
- **New code (KI-064):**
  - an X2 suite driver, which runs trial and replicate training inside the binding, the
    validation sweep with the X2 evaluator, §7 and §6.3 by the frozen pure functions, and
    host guards that sample while jobs run; it is resumable;
  - an X2 latency benchmark (X2_harder_task §4).

  Both go in new files only. A research-integrity test checks that no X2-b code path reaches
  `x2_heldout`.
- **Cost (estimates from the pilot's measured times, local CPU, $0; with the pilot runs
  counted, 20 new runs and 460 new evaluations):**
  - **Training:** 30.5 training-hours summed. Diffusion is 5 × 4.62 = 23.1, BC-RNN 5.6,
    BC-MLP-chunk 1.0 and BC-MLP 0.9.
  - **Evaluations:** 64 container-hours at the measured mean of 503 s. This is biased low:
    the 5k–15k checkpoints were not measured and failing episodes run 500 steps (up to
    819 s), and the 4-container rate was not measured.
  - **Wall time:** about 21 h at the pilot's concurrency. The tuning phase takes 12.8 h:
    three diffusion trials at once set the pace, with 2 evaluation containers while they train
    and 4 after. The replicate phase takes 7.9 h. Fewer containers (item 3) make it longer.
  - **Disk:** about 17.5 GiB: 5.6 GiB of runs and 11.9 GiB of evaluation records at
    26.5 MiB each.
  - **If retrained:** 24 runs and 480 evaluations, adding 6.1 training-hours and
    2.8 container-hours.
- **Labels:** development numbers on validation; X2-b steps no X2 held-out start. X2-c
  (`docs/extensions/X2_FREEZE.md`, the held-out suite once, analysis once) is a later session.

## 6. Exact prompt for Session 16 (X2, phase X2-b; owner decisions first)

```text
Implement RoboForge 3D SESSION 16 ONLY: post-release extension X2, phase X2-b (tuning,
training, selection on validation starts; option A, Can MH), as specified in
docs/extensions/X2_harder_task.md §4, under DECISIONS D-105 and D-115 to D-124 and the rules
of MASTER_SPEC.md.

Begin by reading all of MASTER_SPEC.md, docs/extensions/README.md,
docs/extensions/X2_harder_task.md, docs/STATUS.md, docs/HANDOFF.md (# Session 15 Handoff),
docs/DECISIONS.md (especially D-055, D-061, D-105, D-111 to D-124 and D-124's outcome note),
docs/KNOWN_ISSUES.md (KI-008, KI-035, KI-043, KI-061 to KI-064), docs/EXPERIMENT_LEDGER.md
(Sessions 14 and 15), docs/RESEARCH_PROTOCOL.md (§6, §7), docs/DATA_PROVENANCE.md and
reports/session15/x2/pilot/tables.md, and inspect the actual repository and test state. Do not
blindly trust handoff claims; confirm every prerequisite with executable checks (the X2 data,
suites and pilot records verify against their committed hashes; make check-x2-data and make
check-x2-states pass; runs/x2 is intact).

First ask the owner the decisions of # Session 15 Handoff §5, with the facts measured now (free
disk, swap use, running containers): (1) proceed with X2-b on Can MH, or go back to D-116's
alternatives; (2) count each pilot run as the seed-0 run of the trial whose configuration it
has, or retrain; (3) concurrency limits and the guard; (4) the disk reclaim (the owner runs or
approves it). If the owner stops X2, record the decision and go to the STOP boundary.
Otherwise, for the chosen answers only:
1. Record decisions before acting: Makefile defaults PREFIX ?= s16 and evidence session 16;
   the owner's answers.
2. Write and commit docs/extensions/X2_PREREGISTRATION.md before the first X2-b training: the
   search space (Level 1's trials, D-055/D-061, unless the owner decides otherwise), the equal
   per-family budget, the frozen budget per run (100,000 steps, batch 100, K = 20, every
   checkpoint on all 50 x2_validation starts), the §7 selection and §6.3 configuration rules
   (the frozen pure functions; tuning seed 0, KI-062), the replicate seeds (1 and 2), how the
   pilot runs count, the concurrency and disk limits, and what X2-c will test and report
   (within-operator only; hypotheses and intervals fixed before X2-c).
3. New code in new files only (KI-064): an X2 suite driver (training of trials and replicates
   inside roboforge.x2.binding.x2_bindings, the validation sweep with the X2 evaluator, §7 and
   §6.3 by the frozen pure functions, resumable, the guards: free disk, clean tree, a swap
   guard that samples while jobs run) and an X2 latency benchmark. A research-integrity test
   that no X2-b code path reaches x2_heldout. make freeze-check before and after every code
   change; tests/research_integrity on the new code.
4. Train and select on local CPU from a clean tree, as preregistered; checkpoint selection on
   X2 validation starts only; record the X2 test configuration (selected checkpoints and
   hashes) for X2-c. Measure and report cost against the estimates.

Binding boundaries: Level 1, the RGB study, Level 2, X1, X3 and the X2-a evidence are frozen.
Never rerun make eval-final, make analyze-final, make rgb-heldout,
scripts/analyze_rgb_heldout.py, make pc-heldout, make pc-placement, make level2-analyze,
scripts/analyze_robustness.py, make x1-run, make x1-analyze, make x3-states, make x3-train,
make x3-validate, make x3-check, make x3-run, make x3-analyze, make x2-download,
make x2-states or make x2-pilot, and never rebuild the X2 data with --update-committed. Never
edit reports/session6-10, reports/session13, reports/session14 or reports/session15 results,
the freezes, the X1 or X3 preregistration, LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed
outcome. Never prune or move frozen runs (runs/x1_clamp, runs/x1_checks, runs/x3,
runs/x3_checks, runs/x2 and the s14_* runs included). Never edit a file that make freeze-check
pins. No X2 held-out rollout of any kind in X2-b. No paid or remote compute, push, tag,
release or deployment without explicit owner authorization. No changes to existing data, eval
suites, uv.lock or Python dependencies without a decision. Never uv sync or uv run inside the
image. Use EVIDENCE_SESSION=16; every container job runs via run_sim_cpu.sh with
ROBOFORGE_EVIDENCE_SESSION=16, from a clean tree. Never run the full test suite alongside a
container job (KI-043). Label every estimate as an estimate.

At the SESSION 16 STOP boundary: update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and
EXPERIMENT_LEDGER; run make lint, the full suite alone, make freeze-check,
tests/research_integrity, make release-check; commit verified work as Zackary Holt
<zackholt05@gmail.com> with no AI co-author trailers; inspect git log --format=fuller -n 5;
write the exact prompt for X2-c; STOP. Do not proceed automatically.
```

# Session 14 Handoff (X3)

Do not trust this file blindly. Every claim names its command and evidence
(`logs/session14/`, `reports/session14/`, LEDGER E14-01 to E14-13).

## 1. Session completed
Session 14: post-release extension X3, a separate replication with more training seeds
(`docs/extensions/X3_more_seeds.md`, D-105, D-108, D-111). Status **PASS**. Every X3 result
is labelled "separate replication" and is reported beside, never instead of, the frozen
Level 1 result.

## 2. Completed work
- **Owner decisions.** D-109 (Can only, seeds 3–5, npm cache cleared first); D-113 (pause
  validation while diffusion trains, after swap thrashing); D-114 (disclose 3 evidence files
  with local paths by name); D-115 (Session 15 runs X2).
- **Fixed decisions.** D-110 (Makefile `PREFIX ?= s14`, evidence session 14, X3 targets);
  D-111 (the design; outcome line added); D-112 (an evaluation-only driver finishes validation
  and selection after the suite stopped, KI-062).
- **X3 held-out suite** `can/x3_heldout`: 100 starts from seed block 14,000,000 + i, content
  `9c199e19…`, checked bitwise in the container and committed (`81afa3b`) before any rollout.
- **Preregistration** `docs/extensions/X3_PREREGISTRATION.md` (`fb9f05e`, SHA-256
  `b5e54125…`), committed before the first X3 training run.
- **Training.** 12 new runs (4 families × seeds 3–5), the frozen chosen Can configurations,
  config hash equal to the frozen one except the seed; local CPU, 1 torch thread, from clean
  `fb9f05e`. All complete, 20 checkpoints each, 0 failed (E14-04, E14-08).
- **Validation and selection** on the 50 validation starts only: 240 evaluations, 12
  selections by the frozen §7 rule. The suite selected BC-MLP seed 3 and then stopped at its
  configuration choice (KI-062); `make x3-validate` finished the rest with the suite's own
  commands (D-112): `reports/session14/suite/x3_validation.json` (E14-05, E14-07, E14-09).
- **Engineering check** `make x3-check`: two frozen Can checkpoints reproduced their Session 7
  validation summaries episode by episode (E14-06). Isolation before and after: PASS (E14-10,
  E14-11).
- **X3 evaluations**, once: the 24 selected checkpoints (frozen seeds 0–2, new seeds 3–5) on
  `can/x3_heldout`, 24/24 exit 0, 0 engineering errors (E14-11). **Analysis**, once:
  `reports/session14/x3/` (E14-12).
- **Code** (new files only; `make freeze-check` unchanged before and after every change):
  `src/roboforge/replication/{x3_states,x3,x3_jobs,x3_validation}.py`,
  `scripts/{generate_x3_states,eval_x3,run_x3,check_x3_engineering,analyze_x3,x3_validate}.py`,
  `configs/extensions/x3.yaml`, `configs/extensions/x3/`, `configs/experiments/x3_can.yaml`,
  and tests (`e0b8a4e`, `e9289c6`; E14-05 lists SHA-256). D-114 added 3 names to
  `src/roboforge/release/scan.py` (not freeze-pinned).
- **Records.** KI-009, KI-035, KI-053 updated; KI-062 new; LEDGER E14-01 to E14-13; STATUS;
  `docs/extensions/README.md`.

## 3. Results (separate replication; `reports/session14/x3/tables.md`)
| Contrast (Can) | Frozen, 3 seeds × 50 frozen starts | X3, 6 seeds × 100 X3 starts: bootstrap 95 % | X3 seed-level 95 % | Relation |
|---|---|---|---|---|
| H2 diffusion − BC-MLP | +18.7 pp [+4.7, +32.7], separated (+) | +22.7 [+10.5, +35.0] | [+8.0, +37.3] | holds, robust |
| H2 diffusion − BC-RNN | −0.7 [−8.0, +6.7], not separated | −1.2 [−5.7, +3.0] | [−3.5, +1.2] | unchanged |
| H2-c diffusion − BC-MLP-chunk | −0.7 [−8.7, +7.3], not separated | −1.7 [−6.2, +2.8] | [−4.5, +1.2] | unchanged |
| H3 BC-RNN − BC-MLP | +19.3 [+5.3, +34.0], separated (+) | +23.8 [+12.0, +36.0] | [+9.2, +38.5] | holds, robust |

- Family success on the X3 starts (6 × 100): BC-MLP 72.3 %, BC-RNN 96.2, BC-MLP-chunk 96.7,
  diffusion 95.0 (frozen: 76.7, 96.0, 96.0, 95.3).
- Declared secondary, no readings: on either 3-seed subset (frozen seeds 0–2 on the new
  starts; new seeds 3–5) H2 and H3 separate under the bootstrap but not the seed-level
  interval; only the 6-seed analysis separates them under both. On seeds 3–5 alone, H2-c
  separates (−) under the seed-level interval only ([−5.4, −0.6]).
- The frozen result stands as reported; the two are never pooled. KI-053 stays DOCUMENTED.
- Scope: Can, privileged low-dimensional state, one dataset (PH), R = 1 (KI-028); not Lift,
  not the robustness sets (H4).

## 4. STOP checks
On `227cb26` (code as `bd7c899`; documents only after), host, no container running: `make lint` clean; full suite alone **603 passed** (8:58; the 550 of Session 13 plus 53 Session 14 tests; `logs/session14/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged", also before and after every code change); `tests/research_integrity` 59 passed; `make release-check` PASS (139 disclosed local-path files, D-100 and D-114). No frozen file, freeze, result or executed outcome changed; no frozen held-out suite was rolled out; nothing was pushed, tagged, released or deployed.

## 5. Owner decisions pending
1. Done: the owner chose **X2** for Session 15 (D-115). Session 15 first asks the owner to
   choose option A (Can MH), B (Square PH) or C (Square MH) of `X2_harder_task.md` §2.
2. Whether to publish X1 and X3 (for example a v1.1 with a short addendum to the paper, README
   and Observatory). Not done; nothing was pushed, tagged or deployed (D-105).
3. Optional, not approved: the diffusion side of KI-052 (`clip_sample=False`, about 1.6
   container-hours, an estimate).
4. Host capacity (KI-035): 4 evaluation containers beside 3 diffusion trainings and desktop
   apps thrashed swap on this 16 GB host. Plan X2 training and evaluation so they do not
   overlap at that scale. Disk: about 31 GiB free at the Session 14 STOP (KI-008); the
   Session 14 runs, validation directories and `runs/x3` take 7.1 GiB.

## 6. Exact prompt for Session 15 (X2, phase X2-a; owner, D-115)

```text
Implement RoboForge 3D SESSION 15 ONLY: post-release extension X2, phase X2-a (data and
contract), as specified in docs/extensions/X2_harder_task.md, under DECISIONS D-105 and D-115
and the rules of MASTER_SPEC.md.

Begin by reading all of MASTER_SPEC.md, docs/extensions/README.md,
docs/extensions/X2_harder_task.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md
(especially D-105, D-111 to D-115), docs/KNOWN_ISSUES.md (KI-008, KI-035, KI-043, KI-052,
KI-061, KI-062), docs/EXPERIMENT_LEDGER.md (Sessions 13 and 14), docs/RESEARCH_PROTOCOL.md,
docs/DATA_PROVENANCE.md and docs/SIMULATOR_CONTRACT.md, and inspect the actual repository and
test state. Do not blindly trust handoff claims; confirm every prerequisite with executable
checks.

First ask the owner to choose option A (Can MH, recommended), B (Square PH) or C (Square MH)
of X2_harder_task.md §2, with the facts (file sizes, new infrastructure, free disk measured
now). Then, for the chosen option only:
1. Record decisions before acting: Makefile defaults PREFIX ?= s15 and evidence session 15;
   the data decision (manifest entries with SHA-256 from the pinned revision, verified on
   download).
2. Download and verify; inspect the file (episode count and lengths, masks and filter keys,
   what operator information it actually carries); split by whole episodes, operator-aware if
   it carries operator identity; normalization on the training partition only.
3. For Square: the full Session 0-2 style contract (replay, observation and action conventions,
   success predicate, reset determinism). For Can MH: show that the validated Can contract
   applies (same env_args, controller and action space) before relying on it.
4. Generate an X2 validation suite and an X2 held-out suite on the canonical container from
   fresh seed blocks outside every reserved block in data/eval_states/*/partitions.json and the
   X3 block (14,000,000 + i), named for X2 (never a locked suite's name); check bitwise and
   commit them with their hashes.
5. Headroom pilot on X2 validation starts only (one seed, the frozen Level 1 configurations,
   local CPU): confirm the task is neither saturated nor at zero; measure step time, rollout
   time, memory and disk; propose the X2-b search space, budget, concurrency (KI-035: do not
   run 4 evaluation containers beside diffusion training on this 16 GB host) and disk plan as
   estimates. If a suite plan is used, its tuning seed is 0 (KI-062).

Binding boundaries: Level 1, the RGB study, Level 2, X1 and X3 are frozen. Never rerun make
eval-final, make analyze-final, make rgb-heldout, scripts/analyze_rgb_heldout.py, make
pc-heldout, make pc-placement, make level2-analyze, scripts/analyze_robustness.py, make x1-run,
make x1-analyze, make x3-states, make x3-train, make x3-validate, make x3-check, make x3-run or
make x3-analyze. Never edit reports/session6-10, reports/session13 or reports/session14 results,
the freezes, the X1 or X3 preregistration, LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed
outcome. Never prune or move frozen runs (runs/x1_clamp, runs/x1_checks, runs/x3,
runs/x3_checks and the s14_* runs included). Never edit a file that make freeze-check pins;
run make freeze-check before and after touching any code, and tests/research_integrity/ on new
code. No held-out rollout of any kind in X2-a; the pilot uses X2 validation starts only. No
paid or remote compute, push, tag, release or deployment without explicit owner
authorization. No changes to existing data, eval suites, uv.lock or Python dependencies
without a decision. Never uv sync or uv run inside the image. Use EVIDENCE_SESSION=15; every
container job runs via run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=15, from a clean tree.
Never run the full test suite alongside a container job (KI-043). Label every estimate as an
estimate.

At the SESSION 15 STOP boundary: update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and
EXPERIMENT_LEDGER; run make lint, the full suite alone, make freeze-check,
tests/research_integrity, make release-check; commit verified work as Zackary Holt
<zackholt05@gmail.com> with no AI co-author trailers; inspect git log --format=fuller -n 5;
write the exact prompt for X2-b; STOP. Do not proceed automatically.
```

# Session 13 Handoff (X1)

Do not trust this file blindly. Every claim names its command and evidence
(`logs/session13/`, `reports/session13/x1_clamp/`, LEDGER E13-01 to E13-05).

## 1. Session completed
Session 13: post-release extension X1, the evaluation-only action-clamp test of KI-052
(docs/extensions/X1_action_clamp.md, D-105, D-107). Status **PASS**. Every result is labelled
"exploratory, after the freeze".

## 2. Completed work
- **Decisions.** D-106 (Makefile `PREFIX ?= s13`, evidence session 13, `make x1-run`,
  `make x1-analyze`). D-107 (the design) and its amendment, made before the preregistration.
- **Code, in new files only** (`09d1a90`, analysis amendment in `ab02469`). `make freeze-check`
  passed before and after.
  - `src/roboforge/exploratory/clamp.py`: `ClampedCodec` and `ClampedPolicy` around the frozen
    `LearnedPolicy`, and `evaluate_x1` over the frozen `run_states`.
  - `scripts/eval_x1_clamp.py`, `run_x1_clamp.py`, `check_x1_engineering.py`,
    `analyze_x1_clamp.py`.
  - Tests: `tests/unit/test_x1_clamp.py` and `tests/research_integrity/test_x1_scope.py` (the
    runner refuses the held-out and validation suites).
- **The clamp.** Each normalized BC action is clipped to [−1, 1] before the frozen decoder.
  This is the diffusion sampler's bound bit for bit.
  - It binds on Lift dims 1, 3, 4, 5 and on Can dim 3, plus the upper side of dims 4 and 5.
  - The gripper clamp is a no-op.
- **Engineering checks**, all PASS (`logs/session13/x1_checks/`):
  - exact reproduction of two Session 7 summaries with the clamp disabled;
  - zero effect on diffusion on development starts;
  - the analysis's record checks on real records;
  - isolation before and after.
- **Preregistration** `docs/extensions/X1_PREREGISTRATION.md`, committed in `ab02469` before
  the first reported rollout.
- **45 evaluations** once from `ab02469` (2.60 container-hours). The one analysis is in
  `reports/session13/x1_clamp/`.
- **Records.** KI-052 updated; LEDGER E13-01 to E13-05; STATUS.

## 3. Results (exploratory, after the freeze; `reports/session13/x1_clamp/tables.md`)
- **Primary.** Lift `ood_arm_offset`, drop(diffusion) − drop(clamped BC-MLP-chunk) =
  −40.0 pp [−54.0, −25.3], separated (−). The frozen value is −40.0 [−54.0, −26.0]. Reading R1:
  the bound does not account for all of the frozen H4 gap.
- **Supporting.** drop(BC-MLP-chunk) − drop(clamped) = +0.0 pp [−4.0, +4.0], not separated.
- **Base legs (H2-c).** Clamped − unclamped BC-MLP-chunk on `robustness_base` = +0.0 pp
  [+0.0, +0.0] on both tasks, so H2-c parity stands as measured.
- **The clamp acted, but changed no outcome.** It changed most out-of-distribution BC actions
  on Lift (52–54 % for BC-MLP-chunk, 64–73 % for BC-MLP), but no clamped − unclamped
  difference separated (0 of 15). Every between-family verdict is the frozen one.
- **Reading (BC side).** Bounding BC's executed actions to the training range leaves BC's
  outcomes unchanged, so the H4 gaps are not caused by BC executing out-of-range actions.
- **Not tested (diffusion side).** The sampler clips x̂₀ at every DDIM step, so the bound shapes
  its sampling path. Whether that contributes to diffusion's own robustness needs a
  `clip_sample=False` arm, which was not run (KI-052, half-resolved). The frozen numbers stand.

## 4. STOP checks
On the Session 13 tree (code as `ab02469`; documents only after `c2a9ae6`), host, no container running: `make lint` clean; full suite alone **550 passed** (8:28; the 526 of the baseline plus 24 new X1 tests; `logs/session13/stop_full_suite.log`); `make freeze-check` unchanged ("freeze committed; frozen code unchanged", also before every code change); `tests/research_integrity` 50 passed; `make release-check` PASS (136 disclosed local-path files, D-100). No frozen file, freeze, result or executed outcome changed; nothing was pushed, tagged, released or deployed.

## 5. Owner decisions pending
1. Done: the owner chose **X3** for Session 14 (D-108). Session 14 asks the owner to confirm the
   task(s) and the number of extra seeds.
2. Whether to publish X1 (for example a v1.1 with a short addendum to the paper, README and
   Observatory). Not done; nothing was pushed, tagged or deployed (D-105).
3. Optional, not approved: the diffusion side of KI-052. That is the 6 frozen diffusion
   checkpoints with `clip_sample=False` on the same robustness sets, 15 evaluations; about
   1.6 container-hours (an estimate from the Session 7 times). It is exploratory, after the
   freeze, and needs its own decision and preregistration.
4. The standing optional items of the Session 12 handoff (Modal workspace budget, Docker build
   cache, DNS) are unchanged. Disk: 15 GB free after X1 (KI-008).

## 6. Exact prompts for Session 14
The owner chose X3 (D-108): **use the X3 prompt.** The X2 prompt is kept for a later session;
update its session number and evidence session before using it.

### If the owner chooses X2

```text
Implement RoboForge 3D SESSION 14 ONLY: post-release extension X2, phase X2-a (data and
contract), as specified in docs/extensions/X2_harder_task.md, under DECISIONS D-105 and the
rules of MASTER_SPEC.md.

Begin by reading all of MASTER_SPEC.md, docs/extensions/README.md,
docs/extensions/X2_harder_task.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md
(especially D-105 to D-107), docs/KNOWN_ISSUES.md (KI-008, KI-043, KI-052, KI-061),
docs/EXPERIMENT_LEDGER.md (Session 13), docs/RESEARCH_PROTOCOL.md, docs/DATA_PROVENANCE.md and
docs/SIMULATOR_CONTRACT.md, and inspect the actual repository and test state. Do not blindly
trust handoff claims; confirm every prerequisite with executable checks.

First ask the owner to choose option A (Can MH, recommended), B (Square PH) or C (Square MH)
of X2_harder_task.md §2, with the facts (file sizes, new infrastructure, free disk measured
now). Then, for the chosen option only:
1. Record decisions before acting: Makefile defaults PREFIX ?= s14 and evidence session 14;
   the data decision (manifest entries with SHA-256 from the pinned revision, verified on
   download).
2. Download and verify; inspect the file (episode count and lengths, masks and filter keys,
   what operator information it actually carries); split by whole episodes, operator-aware if
   it carries operator identity; normalization on the training partition only.
3. For Square: the full Session 0-2 style contract (replay, observation and action conventions,
   success predicate, reset determinism). For Can MH: show that the validated Can contract
   applies (same env_args, controller and action space) before relying on it.
4. Generate an X2 validation suite and an X2 held-out suite on the canonical container from
   fresh seed blocks outside every reserved block in data/eval_states/*/partitions.json, named
   for X2 (never the locked suite's name); check bitwise and commit them with their hashes.
5. Headroom pilot on X2 validation starts only (one seed, the frozen Level 1 configurations,
   local CPU): confirm the task is neither saturated nor at zero; measure step time, rollout
   time and disk; propose the X2-b search space, budget and disk plan as estimates.

Binding boundaries: Level 1, the RGB study, Level 2 and X1 are frozen. Never rerun make
eval-final, make analyze-final, make rgb-heldout, scripts/analyze_rgb_heldout.py, make
pc-heldout, make pc-placement, make level2-analyze, scripts/analyze_robustness.py, make x1-run
or make x1-analyze. Never edit reports/session6-10 or reports/session13 results, the freezes,
the X1 preregistration, LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed outcome. Never prune or
move frozen runs (runs/x1_clamp and runs/x1_checks included). Never edit a file that make
freeze-check pins; run make freeze-check before and after touching any code, and
tests/research_integrity/ on new code. No held-out rollout of any kind in X2-a; the pilot uses
X2 validation starts only. No paid or remote compute, push, tag, release or deployment without
explicit owner authorization. No changes to existing data, eval suites, uv.lock or Python
dependencies without a decision. Never uv sync or uv run inside the image. Use
EVIDENCE_SESSION=14; every container job runs via run_sim_cpu.sh with
ROBOFORGE_EVIDENCE_SESSION=14, from a clean tree. Never run the full test suite alongside a
container job (KI-043). Label every estimate as an estimate.

At the SESSION 14 STOP boundary: update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and
EXPERIMENT_LEDGER; run make lint, the full suite alone, make freeze-check,
tests/research_integrity, make release-check; commit verified work as Zackary Holt
<zackholt05@gmail.com> with no AI co-author trailers; inspect git log --format=fuller -n 5;
write the exact prompt for X2-b; STOP. Do not proceed automatically.
```

### If the owner chooses X3 (CHOSEN, D-108: the Session 14 prompt)

```text
Implement RoboForge 3D SESSION 14 ONLY: post-release extension X3, a separate replication
with more training seeds, as specified in docs/extensions/X3_more_seeds.md, under DECISIONS
D-105 and the rules of MASTER_SPEC.md.

Begin by reading all of MASTER_SPEC.md, docs/extensions/README.md,
docs/extensions/X3_more_seeds.md, docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md
(especially D-066, D-095, D-105 to D-107), docs/KNOWN_ISSUES.md (KI-008, KI-043, KI-053,
KI-061), docs/EXPERIMENT_LEDGER.md (Sessions 7 and 13), docs/RESEARCH_PROTOCOL.md §7, §9 and
§14.6 and docs/FINAL_TEST_FREEZE.md, and inspect the actual repository and test state. Do not
blindly trust handoff claims; confirm every prerequisite with executable checks.

First measure free disk and the per-seed cost from the recorded runs, then ask the owner to
confirm the task(s) (recommendation: Can only) and the number of extra seeds, with those facts.
Then:
1. Record decisions before acting: Makefile defaults PREFIX ?= s14 and evidence session 14;
   the X3 design (frozen chosen configurations, new seeds only, the new X3 held-out suite and
   its size, the primary interval and its seed, the contrasts H2, H2-c and H3, side-by-side
   reporting).
2. Generate the X3 held-out suite on the canonical container from a fresh seed block outside
   every reserved block, named for X3 (never the locked suite's name); check it bitwise and
   commit it with its hash before any rollout on it.
3. Write and commit docs/extensions/X3_PREREGISTRATION.md before the first new seed trains.
4. Train the new seeds on local CPU from a clean tree (configuration hash equal to the frozen
   one except the seed); select by the frozen rule on validation starts only.
5. Evaluate every seed (frozen 0-2 and new) on the X3 suite once; analyze once; label every
   result "separate replication", reported beside, never instead of, the frozen ones.

Binding boundaries: Level 1, the RGB study, Level 2 and X1 are frozen. Never rerun make
eval-final, make analyze-final, make rgb-heldout, scripts/analyze_rgb_heldout.py, make
pc-heldout, make pc-placement, make level2-analyze, scripts/analyze_robustness.py, make x1-run
or make x1-analyze. Never edit reports/session6-10 or reports/session13 results, the freezes,
the X1 preregistration, LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed outcome. Never prune or
move frozen runs (runs/x1_clamp and runs/x1_checks included). Never edit a file that make
freeze-check pins; run make freeze-check before and after touching any code, and
tests/research_integrity/ on new code. Training is allowed only for the frozen configurations
with new seeds; no rollout on the frozen held-out suites. No paid or remote compute, push,
tag, release or deployment without explicit owner authorization. No changes to existing data,
eval suites, uv.lock or Python dependencies without a decision. Never uv sync or uv run inside
the image; training runs on the macOS host (KI-061). Use EVIDENCE_SESSION=14; every container
job runs via run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=14, from a clean tree. Never run
the full test suite alongside a container job (KI-043). Label every estimate as an estimate.

At the SESSION 14 STOP boundary (or the end of X3 if it needs a second session): update
STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and EXPERIMENT_LEDGER; run make lint, the full suite
alone, make freeze-check, tests/research_integrity, make release-check; commit verified work as
Zackary Holt <zackholt05@gmail.com> with no AI co-author trailers; inspect git log
--format=fuller -n 5; write the exact prompt for the next session; STOP. Do not proceed
automatically.
```

# Session 12 Handoff (project complete)

Do not trust this file blindly: every claim names its command and evidence
(`docs/FINAL_RELEASE_AUDIT.md`, `logs/session12/`, `reports/session12/`).

## 1. Session completed
Session 12 (MASTER_SPEC Session 12: fresh-clone reproduction, final quality assurance, public
release). Status **PASS**. **This is the last session; there is no next-session prompt.**

## 2. Completed work
- **Decisions D-097 to D-104**, the owner's choices asked once with the facts: public repo, tag
  `v1.0.0` and GitHub release (D-099); GitHub Pages (D-102); CI on push (D-101); local paths
  disclosed, no history rewrite (D-100).
- **Fresh-clone reproduction** (D-098): full suite 519 passed and 7 skipped (each with its
  reason); container and host smoke 33/33; download links and SHA-256; data rebuild equal;
  `make paper` byte-identical (`c648e753…`); site build, vitest 49/49, E2E 7/7; a frozen
  checkpoint hash-checked against the freeze and loaded. It found and fixed two test-only
  defects (KI-058, KI-059).
- **Release tooling** (`src/roboforge/release/`, `scripts/release_check.py`,
  `readme_findings.py`, `readme_preview.py`, `pages_bundle.py`, `license_inventory.py`) and
  `make release-check`, `readme-findings`, `readme-preview`, `pages-bundle`.
- **README** per MASTER_SPEC §11.2 with generated findings and setup blocks and a preview from
  recorded renders; `CITATION.cff` v1.0.0; `docs/REPRODUCIBILITY.md`, `THIRD_PARTY_NOTICES.md`,
  site credits; `docs/FINAL_RELEASE_AUDIT.md`.
- **CI** (`.github/workflows/`): python-tests, research-integrity, web-build, on Linux aarch64.
  The first runs found a triton segfault whose root cause (Mesa's LLVM in the global symbol
  scope) was isolated in the canonical container and fixed in the test harness (KI-061).
- **Publication** (§5): public repository, GitHub Pages Observatory verified file by file.

## 3. Results (nothing here is a new experimental result)
The frozen findings are the generated block of `README.md` and `docs/FINAL_RELEASE_AUDIT.md` §2
(from `paper/generated/numbers.tex`): Can diffusion − BC-MLP separated under the preregistered
bootstrap but fragile under the seed-level check; H2-c not separated on either task; BC-RNN −
BC-MLP separated on both; diffusion costs far more to train and run; the RGB comparison not
separated at the Lift ceiling; H5 separated in the negative direction (point clouds generalized
worse to shifted cubes).

## 4. STOP checks
On `b4bd070` (release candidate), host: `make lint` clean; full suite alone
526 passed (8:32, `logs/session12/stop_full_suite.log`); `make freeze-check` unchanged; both
gates open; `make paper-check` pass; `make release-check` PASS; `make web-build`, `web-test`
49/49, `web-e2e` 7/7. After the test-harness fix for KI-061 (`ab17fb2`, `tests/conftest.py`
only): `make lint` clean; full suite alone 526 passed (8:20,
`logs/session12/stop_full_suite_2.log`); `make freeze-check` unchanged; `tests/research_integrity`
45 passed; `make release-check` PASS. CI on `ee1e574` (fresh checkouts, Linux aarch64): all
three workflows pass (publication record).

## 5. Publication record

Each row was verified before it was written (owner authorization D-099, D-101, D-102).

| Step | Verified state | Evidence |
|---|---|---|
| Repository | public, `https://github.com/zackholt05-tech/roboforge`; `main` pushed (first `b4bd070`, then each later commit, checked with `git ls-remote`) | `gh repo view`; audit §7 |
| CI, Linux x86_64, `b4bd070` | research-integrity pass (run 42755805851); python-tests fail, triton segfault (42755805869); web-build fail, mesh check 7.45e-9 m (42755806070) | KI-061 |
| CI, Linux aarch64, `e11ad04` | research-integrity pass (42756520065); web-build pass (42756520080); python-tests fail, same triton segfault (42756519847) | KI-061 root cause |
| CI, Linux aarch64, `ee1e574` (after the harness fix) | python-tests pass (42758938590: lint; 303 passed, 12 skipped: 9 without the processed Lift data, 2 without `runs/`, 1 needs MPS); research-integrity pass (42758938394); web-build pass (42758938356) | D-101 amendment 2 |
| Observatory | GitHub Pages from branch `gh-pages`, orphan commit `c14e041` (Zackary Holt), staged by `make pages-bundle` from clean `b4bd070`; build `built`; `https://zackholt05-tech.github.io/roboforge/` | `gh api …/pages` |
| Live site check | all 186 files listed in `deploy.json` fetched from the live URL with equal SHA-256; the site's E2E run against the live URL 7/7 | `logs/session12/pages_live_e2e.log` |
| External links after publication | 19/19 answered with HTTP < 400 | `reports/session12/release_check_network.json` |
| CI, Linux aarch64, `7b394fa` (the tagged commit) | python-tests pass (42760766015), research-integrity pass (42760766276), web-build pass (42760766117) | |
| Tag `v1.0.0` | annotated tag object `82f3219` on `7b394fa`, tagger Zackary Holt <zackholt05@gmail.com>, no trailer; pushed (`git ls-remote`) | `git cat-file -p v1.0.0` |
| GitHub release | `https://github.com/zackholt05-tech/roboforge/releases/tag/v1.0.0`, published 2026-11-10 05:24 UTC, not a draft or prerelease; notes written by hand (no generated notes, no byline); asset `main.pdf` (878,204 bytes), GitHub digest and the SHA-256 of the downloaded copy both `c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f`, equal to the committed `paper/main.pdf` | `gh release view`, `gh release download` |


## 6. Outstanding items (none blocks the release)
- Open KIs: KI-001/KI-033 (EGL vs OSMesa on Modal), KI-008/KI-010 (disk, image size), KI-011,
  KI-021, KI-023, KI-024, KI-028, KI-034, KI-035, KI-042; KI-061 partial (any Linux process that
  imports mujoco with OSMesa before its first torch optimizer crashes; workaround recorded);
  KI-052 and KI-053 documented; KI-060 accepted.
- Not run, by decision: the KI-052 clamp test (needs its own decision and the label
  "exploratory, after the freeze"); DiT, GMM-BC, Square (D-096).
- Optional owner actions: a Modal workspace budget; pruning the Docker build cache (KI-008).

# Session 11 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 11 (MASTER_SPEC §11 and Session 11: research paper, reproducibility and independent
critical audit). Status **PASS, local only, $0**.
- **Paper:** `paper/main.pdf` (23 pages, SHA-256
  `c648e7532e7112cd8bcb3b4e0f118342d6eb97dffd94d3bd1e5448c81b50939f`), a local build, labelled
  "not peer reviewed". Every number in the prose is a macro generated from hash-pinned frozen
  files; a lint fails on any typed number or placeholder.
- **Audit:** four fresh-context passes (`docs/AUDIT.md`). Each finding has a disposition: an
  engineering correction or a limitation disclosed in the paper. No frozen file changed.
- **No training, no evaluation, no paid or remote compute, no hosting.**

## 2. Completed work
- **Decisions:** D-092 (Makefile defaults s11; `OBS_OUT` pinned to Session 10), D-093 (Tectonic
  toolchain, reproducible PDF, no new dependency), D-094 (generator contract: pinned sources,
  provenance index, lint), D-095 (seed-level sensitivity analysis, not preregistered, beside
  every frozen interval; censored-length success/speed split), D-096 (audit dispositions;
  declared secondary metrics reported post hoc from hash-chained summaries; the A-6 clamp test
  not run; DiT, GMM-BC and Square not run, with the reason).
- **Generator** (`src/roboforge/paper/`, `scripts/paper_assets.py`): `sources` (30 pinned
  files, `chained_json` for files whose hash a pinned file records), `numbers` (1,503
  macros), `tables` (17), `figures` (6, plus 3 renders located by SHA-256), `sensitivity`,
  `secondary`, `lint`, `build` (byte-for-byte `check`).
- **Bibliography:** `paper/references.bib` (31 entries from DOI and arXiv metadata),
  `scripts/verify_bibliography.py`, `paper/bibliography_check.json` (31/31 resolved).
- **Paper:** `paper/main.tex`, `paper/sections/*.tex` (abstract, 10 sections, reproducibility
  appendix, additional tables).
- **Audit fixes:** run-once guard in `scripts/analyze_robustness.py` (B-12, with a test);
  existence-before-hash in `scripts/analyze_level2.py` (B-13); RGB preemption footnote (C-6);
  D-086 and E9-11 record errors annotated (C-20); every paper claim corrected per audit D.
- **Docs:** `docs/AUDIT.md` (new), `docs/REPRODUCIBILITY.md` (rewritten for the current state),
  KNOWN_ISSUES KI-052–KI-057 and the missing summary rows KI-044–KI-051, ledger E11-01–E11-07.

## 3. Files created or changed (Session 11)
- New: `docs/AUDIT.md`; `paper/` (`main.tex`, `sections/`, `references.bib`,
  `bibliography_check.json`, `generated/`, `main.pdf`); `src/roboforge/paper/`;
  `scripts/paper_assets.py`, `scripts/verify_bibliography.py`;
  `tests/unit/test_paper_assets.py`, `tests/unit/test_paper_tex.py`.
- Changed: `Makefile` (defaults, `paper-assets`, `paper-check`, `paper-bibcheck`, `paper`),
  `scripts/analyze_robustness.py`, `scripts/analyze_level2.py`,
  `tests/unit/test_analyze_robustness.py`, `docs/{DECISIONS,KNOWN_ISSUES,EXPERIMENT_LEDGER,
  REPRODUCIBILITY,STATUS,HANDOFF}.md`.
- Unchanged: every frozen result, freeze, `LEVEL1_COMPLETE`, `LEVEL2_GO_NO_GO`, data, suites,
  `uv.lock`, `pyproject.toml`, `website/`.

## 4. Executed commands and results
| Command | Result |
|---|---|
| `make lint` | All checks passed |
| `make test` (alone, no container job) | **513 passed** (520 s) |
| `make freeze-check` (before and after the two analysis-script fixes, and at STOP) | frozen code unchanged |
| `uv run --frozen python scripts/run_rgb_heldout.py gate` / `run_pc_heldout.py gate` | `"open": true, "problems": []` (both) |
| `make paper-assets`, `make paper-check` | 35 files; matches a fresh build; lint clean |
| `make paper-bibcheck` (network) | 31/31 resolved |
| `make paper` (twice) | no undefined reference or citation, no overfull box; identical SHA-256 |
| `make web-build`, `make web-test` | within budgets; vitest 49/49 |

## 5. Key results (nothing here is a new experimental result)
- The paper restates the frozen results, with the post-hoc D-095 and D-096 analyses labelled.
- Audit HIGH findings: A-6 (the diffusion sampler bounds actions to the training range; BC
  heads are not bounded; a confound for H2-c and H4, KI-052); B-1/B-2 (Can H2, Can H3 and six
  robustness separations depend on the interval method, KI-053); D-1 ("not separated" had
  been written as equality in the draft; corrected).
- Spend reconciles launch by launch to $10.61436 over 24 launches (audit C-1).

## 6. Decisions and issues
- D-092–D-096 (above).
- New issues: KI-052 (MEDIUM, documented), KI-053 (MEDIUM, documented), KI-054 (MEDIUM, fixed by
  reporting), KI-055 (LOW, fixed), KI-056 (LOW, documented), KI-057 (LOW, documented).

## 7. Budget
$0 this session (local only). Total billed stays $10.61; no approval is open.

## 8. Outstanding items (none blocks Session 12)
- **KI-052 clamp test:** an evaluation-only exploratory test of BC with outputs clamped to the
  training range would resolve the A-6 confound. It is a new post-freeze evaluation: it needs
  its own decision, a clean tree and the label "exploratory, after the freeze", and it is not
  part of MASTER_SPEC Session 12.
- **No remote:** the repository has no git remote. Publication needs owner authorization and
  credentials (MASTER_SPEC Session 12).
- **Tectonic bundle:** a fresh machine needs one networked Tectonic build before
  `make paper` works offline (D-093; `docs/REPRODUCIBILITY.md`).
- **Open from earlier sessions:** KI-001/KI-033 (EGL vs OSMesa), KI-008 (disk), KI-034 (Lift
  saturation) and the other open KIs in STATUS.

## 9. What Session 12 can use
- `docs/REPRODUCIBILITY.md` (current commands), `docs/AUDIT.md` (findings and dispositions),
  `paper/main.pdf` and `make paper-check` (a reproducible table and figure build from frozen
  files), the Observatory site (`make web-build`; no hosting).
- A representative reproduction target: `make paper-check` regenerates every table and figure
  from the committed frozen files and compares them byte for byte.

## 10. Exact starting instructions for Session 12
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 12 ONLY from MASTER_SPEC.md (fresh-clone reproduction, final
quality assurance, public release). Begin by reading all of MASTER_SPEC.md (especially §11,
§11.2, §11.3 and the Session 12 section), docs/STATUS.md, docs/HANDOFF.md
"# Session 11 Handoff", docs/AUDIT.md, docs/DECISIONS.md (D-013, D-090, D-092–D-096),
docs/KNOWN_ISSUES.md (every open item; KI-052–KI-057), docs/EXPERIMENT_LEDGER.md (Sessions
10–11 and the approvals table: no approval is open), docs/REPRODUCIBILITY.md, README.md,
CITATION.cff, LICENSE and paper/main.tex.

Inspect the real state first:
- git log --format=fuller -n 8; git status; git remote -v (Session 11 found no remote);
- make lint; make freeze-check; make paper-check;
- `uv run --frozen python scripts/run_rgb_heldout.py gate` and
  `uv run --frozen python scripts/run_pc_heldout.py gate` (both must print open);
- never run the full test suite alongside a container job (KI-043).

1. Record decisions before acting: Session 12 Makefile defaults (PREFIX ?= s12, evidence
   session 12; the D-092 pattern); the fresh-clone environment (for example a clean clone in a
   temporary directory and the canonical container); what "release" means here. Ask the owner
   before any public repository, push, tag, release or hosting, and never claim one happened
   without verifying it.
2. Fresh-clone reproduction in a separate clean directory: install the pinned environment
   (uv sync --frozen on the host only, never inside the image), run the tests, a small
   simulator smoke, `make paper-check` (regenerates every paper table and figure from the
   frozen files), `make paper` (note the one-time networked Tectonic bundle build, D-093) and
   `make web-build`. Record every command and result. Checkpoints are not in git; state that.
3. Verify data download links and exact commands, and scan for dead placeholders, private
   paths, secrets, copyrighted dataset copies and broken citations.
4. Verify every commit's author and committer name and email, the absence of AI co-author
   trailers in the whole history, and CITATION.cff (Zackary Holt as project author). Do not
   rewrite history without owner approval.
5. Run the full test matrix: Python (alone), simulator, research-integrity, website (vitest,
   E2E), lint, type checks if configured, a security and license scan. Document any skipped
   test with its precise reason.
6. README per MASTER_SPEC §11.2 from real artifacts only: key findings from the frozen tables
   (with the audit's caveats: the Can H2 separation is method-dependent, H2-c not separated,
   the diffusion action bound KI-052), links to the paper (it exists, labelled not peer
   reviewed) and no Observatory link unless a deployed site exists.
7. Write docs/FINAL_RELEASE_AUDIT.md: pass/fail per check, the actual frozen results, executed
   commands, release hashes (paper/main.pdf SHA-256 and the release commit) and outstanding
   limitations.
8. STOP (project complete): update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and
   EXPERIMENT_LEDGER; run lint, the full test suite (alone), freeze-check, both gates,
   make paper-check, make web-build and make web-test; commit verified work as Zackary Holt
   <zackholt05@gmail.com> with no AI co-author trailers; inspect git log --format=fuller -n 5;
   write the project completion summary (no next-session prompt); STOP. Do not start new
   phases.

Boundaries (binding):
- Level 1, the RGB study and Level 2 are frozen. Never rerun make eval-final, make
  analyze-final, make rgb-heldout, scripts/analyze_rgb_heldout.py, make pc-heldout, make
  pc-placement, make level2-analyze or scripts/analyze_robustness.py. Never edit
  reports/session6-10 results, the freezes, LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed
  outcome. Never prune or move frozen runs.
- The paper's numbers come only from `make paper-assets`; never type a number into
  paper/sections/. Any change to paper/ is followed by make paper-check and make paper.
- Run make freeze-check before and after touching any simulator, evaluation or analysis code,
  and tests/research_integrity/ on new code.
- No training. No new evaluation (the KI-052 clamp test is out of scope unless the owner asks
  for it and a decision is recorded first). No paid or remote compute, push or hosting without
  explicit owner authorization (and an approval row for paid compute).
- No changes to data, eval suites, uv.lock or Python dependencies without a decision.
- Never uv sync or uv run inside the image.
- Use EVIDENCE_SESSION=12. Any container job runs via run_sim_cpu.sh with
  ROBOFORGE_EVIDENCE_SESSION=12, from a clean tree.
- Label every estimate as an estimate.
```

# Session 10 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 10 (MASTER_SPEC §8 and Session 10: the RoboForge Observatory). Status **PASS, local
only**.
- **What it is:** a static site in `website/` that tells the project's story from frozen
  evidence. It covers the recorded rollouts, the diffusion sampler and the findings of
  Level 1, the RGB study and Level 2.
- **Data:**
  - every rollout is a recorded evaluation episode, re-rendered from its saved simulator
    states and never re-run;
  - the only new policy calls are 3 denoising traces on validation starts, replayed bitwise
    and labelled as an export.
- **Checks:**
  - the viewer matches MuJoCo's own renders on 18 reference frames;
  - every number on the site is copied from a frozen results file.
- **Hosting:** the owner chose **no hosting** (D-090). **No deployed or staging URL exists.**
- **Cost:** $0, local CPU and the canonical container only.

## 2. Completed work
- **Decisions:**
  - D-087: Makefile defaults for Session 10 (`PREFIX ?= s10`, evidence session 10).
  - D-088: the web stack, a separate `website/` package with its own lockfile. React 19,
    React Three Fiber 9, three 0.187, zod, Vite, Vitest, Playwright. `uv.lock` and
    `pyproject.toml` are unchanged.
  - D-089: selection rules and provenance, applied by `make observatory-select` (37 clips).
  - D-090: the owner's choices: no hosting; robot meshes extracted at build time and never
    committed.
  - D-091: viewer-verification thresholds, reference masks drawn without sites, and one
    export from one commit. It has an amendment (frozen renderer restored).
- **Export (E10-05):** 37 clips from clean `b5fc2ee` on the canonical container, 3,548 s.
  - 28 Level 1 held-out and 6 Level 2 placement records;
  - recomputed body poses equal the recorded poses to 0.0 m;
  - 3 validation denoising traces with clone-simulated candidate paths;
  - point clouds and RGB policy views at each decision frame;
  - 18 reference frames.
- **Findings (E10-06):** `findings.json` copies 8 frozen sources verbatim, each with its
  SHA-256. Section F was first built without the Level 1 robustness and ablation analysis.
  That was caught before STOP: the analysis is now a source, and F shows it, labelled
  "validation regime, not held-out" (`8679aa9`, `c73a080`, `733d840`).
- **Site, sections A–H of MASTER_SPEC §8.2:**
  - **A. Hero:** a recorded Lift rollout with its caption.
  - **B. Why diffusion:** an illustrative schematic kept apart from the recorded sampler
    tensors.
  - **C. The lab:** task and policy selectors, scrubber, camera presets.
  - **D. Head to head:** two seed-matched recorded rollouts from the same start, on a shared
    clock, each with its outcome, steps, end-effector trail and recorded control commands.
  - **E. Inside the policy:** scheduler slider; normalized chunks and environment-space
    actions; observation window; clone-simulated candidate paths; 16 extra seeds; the
    measured path spread.
  - **F. Findings:** held-out success with denominators and intervals, paired differences,
    per-seed rings, outcome strips, failure taxonomy, latency against success.
    Robustness (H4) and ablations A1–A3 are kept apart.
  - **G. Beyond pixels:** the RGB study and Level 2 (H5 −41.3 points [−54.0, −28.0],
    separated (−)), the placement maps, the KI-049 erratum and the calibrated overlays.
  - **H. Methods:** approximations, verification table, references, credits.
- **Quality:**
  - accessibility (text alternatives on every chart, keyboard transport, skip link);
  - reduced motion;
  - phone layout;
  - a no-WebGL fallback to the MuJoCo video;
  - explicit errors for missing artifacts;
  - canvases released off screen;
  - asset budgets enforced by `website/scripts/budget.mjs`.
- **Provenance ledger:** `docs/VISUAL_PROVENANCE.md`, generated by `make visual-provenance`.

## 3. Files created or changed (Session 10)
- **Python** (`src/roboforge/viz/`):
  - `obs_select.py`: selection rules;
  - `obs_scene.py`: recomputed poses and provenance;
  - `obs_media.py`: video, reference frames, segmentation without sites;
  - `obs_branch.py`: clone simulation;
  - `obs_denoise.py`: sampler traces;
  - `obs_findings.py`: the verbatim copy.
- **Scripts:** `scripts/export_observatory.py` and `scripts/visual_provenance.py`.
- **Tests:**
  - `tests/unit/test_observatory.py` and `tests/unit/test_visual_provenance.py`;
  - `tests/simulator/test_observatory_export.py` and
    `tests/simulator/test_observatory_reference_masks.py`.
- **Site:** `website/`:
  - `src/` covers sections, charts, three.js viewer, schemas, the verification page and
    tests;
  - also `e2e/`, `scripts/budget.mjs`, `package.json` and `package-lock.json`.
- **Evidence:** `reports/session10/observatory/`:
  - `selection.json`, `index.json`, `meshes.json`, `findings.json`,
    `viewer_verification.json`;
  - `clips/`, `media/`, `denoise/`, `clouds/`, `policy_views/`, `verification/` (with
    `viewer_montage.jpg`);
  - `site_screens/01–04`.
- **Logs:** `logs/session10/` (the export log, the stopped partial export's log,
  development logs).
- **Docs:** README (Observatory section), `Makefile` (observatory and web targets),
  `.gitignore`. The STOP commit adds DECISIONS D-087–D-091, KNOWN_ISSUES KI-050,
  `EXPERIMENT_LEDGER` Session 10, STATUS and this handoff.

## 4. Executed commands and results
| Command | Where | Result |
|---|---|---|
| `make observatory-select` | host, `d12729f` | 37 clips; reproduces the Level 1 freeze's example episodes |
| `make observatory-export` | container, clean `5f398d7` | **stopped** at clip 19/37 (KI-050); partial output discarded |
| `make observatory-export` | container, clean `b5fc2ee` | 37/37 clips, 155 files, 3,548 s; recompute 0.0 m |
| `make observatory-findings` | host, clean `8679aa9` | 8 frozen sources (first run `ebfd0c3`: 7) |
| `make web-data web-assets` | host | 161 data files hash-checked; 61 meshes, 3.12 MB, host = container to 0.0 m |
| `make web-build` | host | budgets pass: JS 382 KB gzip, first view 3.94 MB, all data 18.5 MB |
| `make web-test` | host | vitest 49/49 |
| `make web-verify` | host, headless Chrome | 18/18 frames pass: robot IoU ≥ 0.9998, object IoU ≥ 0.9970, centroid ≤ 0.04 px |
| `make web-e2e` | host | 7/7 |
| dev re-render (`--dev-dir observatory_dev4`) | container, restored renderer | 43/43 files identical to the committed export (E10-08) |
| `make lint`, `make freeze-check`, both gates | host | clean; frozen code unchanged; RGB and Level 2 gates open |
| `make test` (alone) | host | `733d840`: 491 passed, 1 failed (KI-051); `4348244`: **492 passed** (500.8 s) |

## 5. Key results (nothing here is a new experimental result)
- **Viewer correctness:** silhouettes through the exported MuJoCo cameras reproduce MuJoCo's
  segmentation. The per-frame table is in `viewer_verification.json` and the montage in
  `verification/viewer_montage.jpg`.
- **Denoising traces:** Lift `lift_validation_0000` (6 decisions, success at step 47). Can
  `can_validation_0000` (13 decisions, success at step 103) and `can_validation_0003`
  (50 decisions, failure at 400 steps). Every replayed chunk equals the recorded action.
- **Path spread** on Lift's first decision, measured in the browser from the exported paths:
  RMS 0.54 cm after 8 steps (largest pair 1.57 cm), 0.51 cm after 16 steps (1.41 cm).

## 6. Decisions and issues
- D-087–D-091 are listed in §2. D-089 is amended (the robustness analysis is a source)
  and D-091 is amended (the frozen renderer is restored).
- **KI-050 (fixed):**
  - the first export stopped on the random baseline's null checkpoint;
  - the reference masks counted robosuite's transparent sites;
  - the first fix edited the frozen renderer. `make freeze-check` caught it, the file was
    restored, and the fix moved to `viz/obs_media`.
- **KI-051 (fixed):** the research-integrity lock test failed at STOP. `obs_select.py`
  wrote the locked suite's partition label as a literal. The label is now read through
  `eval_states`, and the selection reproduces exactly (`4348244`).
- **Rules:**
  - run `make freeze-check` before touching any simulator or evaluation code;
  - run `tests/research_integrity/` on new code before any evidence run.
- **Note, not an issue:** a browser running a WebGL loop on the same host slowed the
  container export about 7× (inferred, not measured). Park the browser during long container
  runs.

## 7. Budget
$0 this session (local only, no hosting). The total billed stays $10.61; no approval is open.

## 8. Outstanding items (none blocks Session 11)
- **No hosting.** Publishing `website/dist` needs a new approval row and a choice of host.
- **Meshes are not in git (D-090).** A fresh clone runs `make web-assets`, which needs the
  pinned robosuite on the host, before the 3D viewer shows meshes. Without them the viewer
  draws bounding boxes and says so.
- **Console warning:** three 0.187 logs a deprecation warning from React Three Fiber's use of
  `THREE.Clock`. It is a library warning, not an error; E2E fails on errors only.
- **Manifest links:** without hosting, "persistent links to manifests" (§8.2 F) take the form
  of repository paths with SHA-256 hashes.
- **Open from earlier sessions:** KI-001/KI-033 (EGL vs OSMesa), KI-008 (disk: about 17 GB
  free), KI-034 (Lift saturation) and the other open KIs listed in STATUS.

## 9. What Session 11 can use
- **Tables and figures:** every frozen results file:
  - `reports/session7/results/` (held-out) and `reports/session7/robustness/` (H4, A1–A3);
  - `reports/session8/rgb_results/`;
  - `reports/session9/level2_results/` (with `ERRATA.md`);
  - each has run manifests and hashes.
- **Paper figures:** `reports/session10/observatory/findings.json` is a hash-cited
  convenience copy. The paper must cite the original files.
- **Genuine visuals for the paper:**
  - `site_screens/`;
  - `verification/viewer_montage.jpg`;
  - the MuJoCo videos in `media/`;
  - the Session 9 overlays.
- **Tooling:** Tectonic is installed at `/opt/homebrew/bin/tectonic`; `pdflatex` and
  `latexmk` are not (checked 2026-11-09).

## 10. Exact starting instructions for Session 11
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 11 ONLY from MASTER_SPEC.md (research paper, reproducibility
and independent critical audit). Begin by reading all of MASTER_SPEC.md (especially §11,
§11.1–11.3 and the Session 11 section), docs/STATUS.md, docs/HANDOFF.md
"# Session 10 Handoff", docs/DECISIONS.md (D-063, D-081, D-083–D-091), docs/KNOWN_ISSUES.md
(every open item; KI-034, KI-045–KI-051), docs/EXPERIMENT_LEDGER.md (Sessions 6–10 and the
approvals table: no approval is open), docs/RESEARCH_PROTOCOL.md, docs/REPRODUCIBILITY.md,
docs/COMPUTE_PLAN.md, docs/LEVEL1_COMPLETE.md, docs/FINAL_TEST_FREEZE.md,
docs/RGB_EXPERIMENT_FREEZE.md, docs/LEVEL2_EXPERIMENT_FREEZE.md, docs/LEVEL2_GO_NO_GO.md,
docs/VISUAL_PROVENANCE.md, reports/session7/results/tables.md,
reports/session7/robustness/tables.md, reports/session8/rgb_results/tables.md and
reports/session9/level2_results/{tables.md,ERRATA.md}.

Inspect the real state first:
- git log --format=fuller -n 8; git status;
- make lint; make freeze-check;
- `uv run --frozen python scripts/run_rgb_heldout.py gate` and
  `uv run --frozen python scripts/run_pc_heldout.py gate` (both must print open);
- never run the full test suite alongside a container job (KI-043).

1. Record decisions before adding anything:
   - Session 11 Makefile defaults (PREFIX ?= s11, evidence session 11; the D-082 pattern);
   - the paper toolchain. Session 10 found Tectonic at /opt/homebrew/bin/tectonic and no
     pdflatex or latexmk: verify this yourself. No new Python or npm dependency without a
     decision; uv.lock and pyproject stay unchanged unless decided.
2. Generate every table and figure directly from the frozen run manifests and results files,
   by a script and a make target, with tests. Each output records its source paths and
   SHA-256 hashes. Cover:
   - core performance;
   - paired differences with CIs;
   - model and seed dispersion;
   - OOD results (H4, H5);
   - the action-horizon and denoising ablations (A1–A3);
   - real hardware, cost and latency (with the KI-049 erratum beside the row it corrects);
   - the failure taxonomy;
   - the RGB and point-cloud panels.
   Cite the original reports/session6-9 files, not the Observatory's findings.json copy.
3. Write the LaTeX paper in paper/ with the structure of MASTER_SPEC §11.1:
   - real methods and results, no placeholders and no typed numbers (generated macros or
     tables only);
   - quantitative limitations, a reproducibility appendix and a verified third-party
     bibliography (check that every DOI or arXiv id resolves);
   - the single project author, Zackary Holt;
   - claims within MASTER_SPEC §11.3: no novelty or significance beyond what the
     preregistered analyses show; "DP3-inspired", never a DP3 reproduction.
4. Independent critical audit:
   - run it as a separate pass that reads the code and the frozen files, not this handoff's
     conclusions;
   - check simulator/controller alignment, evaluation seed pairing, split leakage, privilege
     mismatch, invalid confidence intervals, dataset reuse, GPU expenditure, missing null
     results, and incorrectly claimed novelty or statistical significance;
   - write the findings to docs/ (for example docs/AUDIT.md), with severity and evidence;
   - each finding gets an engineering correction or a transparent limitation. A material
     post-freeze correction to a test protocol is labelled a corrected/replication analysis
     with full provenance, needs a recorded decision before any new evaluation, and never
     replaces the original result.
5. Build paper/main.pdf with the verified toolchain. If none works, document the exact
   compilation commands and do not claim a PDF exists.
6. STOP:
   - update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and EXPERIMENT_LEDGER;
   - run lint, the full test suite (alone), freeze-check, both gates, and make web-build and
     make web-test (the site must still build);
   - commit verified work as Zackary Holt <zackholt05@gmail.com>, with no AI co-author
     trailers;
   - inspect git log --format=fuller -n 5;
   - write the exact Session 12 prompt, then STOP.

Boundaries (binding):
- Level 1, the RGB study and Level 2 are frozen. Never rerun make eval-final, make
  analyze-final, make rgb-heldout, scripts/analyze_rgb_heldout.py, make pc-heldout, make
  pc-placement or make level2-analyze. Never edit reports/session6-10 results, the freezes,
  LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed outcome. Never prune or move frozen runs.
- Run make freeze-check before and after touching any simulator, evaluation or analysis code
  (D-091 amendment), and tests/research_integrity/ on new code before any evidence run
  (KI-051).
- No training. No paid or remote compute (including hosting) without an approval row.
- No changes to data, eval suites, uv.lock or Python dependencies without a decision.
- Never uv sync or uv run inside the image.
- Use EVIDENCE_SESSION=11. Any container job runs via run_sim_cpu.sh with
  ROBOFORGE_EVIDENCE_SESSION=11, from a clean tree.
- Label every estimate as an estimate.
```

# Session 9 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 9 (MASTER_SPEC §7.3 and Session 9: Level 2, explicit 3D perception, DP3-inspired).
Status **PASS, GO path**.
- **Feasibility:** depth from the RGB arm's two cameras, calibrated to metric world-frame
  points, verified numerically and visually on the canonical container.
- **Decision:** the owner chose GO on the L4 (A-04, $9 cap) and made the placement test (H5)
  primary.
- **Training:** a DP3-inspired point-cloud diffusion policy, 3 seeds, under a committed freeze
  (`b93c50c`). **$2.37 billed.**
- **Evaluation:** validated and selected by §7; the 12 preregistered placement evaluations
  for both arms; the held-out test once behind a new gate (D-086); the analysis once.
- **Level 1 and the RGB study:** frozen and untouched.

**Result (Lift):**
- `ood_cube_shift`: point cloud **56.7 %** against RGB **98.0 %**. **H5: −41.3 points
  [−54.0, −28.0], separated (−).**
- Base and held-out: 100 % for both arms, not separated.
- A negative result for H5, reported as such.

## 2. Completed work
- **D-082:** Makefile defaults for Session 9 (`PREFIX ?= s9`, evidence session 9).
- **D-083:** the depth and point-cloud path (`src/roboforge/data/pointcloud.py`):
  - sites hidden;
  - reversed-Z conversion;
  - pixel-centre back-projection;
  - fixed box crop;
  - FPS to 512 points;
  - sensor-like privilege.
- **Feasibility:** `scripts/pc_feasibility.py`, with diagnostics in `pc_checks`:
  - calibration, invariants, coverage, cost and overlays;
  - the superseded `a20bfe3` run disclosed (planarity condition added).
- **D-084:** the point-cloud dataset (9,666 frames, 56.3 MB), built on the container and
  installed.
- **Arm:**
  - `models/point_encoder.py` (DP3-inspired);
  - `train/pointcloud.py` (`PCTrainer` subclasses the frozen `RGBTrainer`);
  - the config matched to the RGB diffusion arm.
- **Compute:**
  - `compute/pc_train.py`, `train_gpu_pc` (L4) and `scripts/pc_train_queue.py`, with a
    step-time stop rule;
  - **D-085:** detached, single-flight launches that re-attach by call ID
    (`compute/launch_wait.py`).
- **Evaluation:**
  - `eval/pointcloud.py` and `scripts/eval_pc.py` (validate, select, placement);
  - **D-086:** `eval/pc_heldout.py`, `scripts/run_pc_heldout.py` and the guarded test.
- **Analysis:** `scripts/analyze_level2.py`, run once; erratum KI-049.
- **Protocol:** RESEARCH_PROTOCOL §2 (H5), §13 amendment, §15; `docs/LEVEL2_EXPERIMENT_FREEZE.md`.

## 3. Files created or changed (Session 9)
- **Docs:**
  - new: `docs/LEVEL2_GO_NO_GO.md`, `docs/LEVEL2_EXPERIMENT_FREEZE.md`;
  - updated: `docs/{STATUS,HANDOFF,DECISIONS,KNOWN_ISSUES,EXPERIMENT_LEDGER,COMPUTE_PLAN,RESEARCH_PROTOCOL}.md`
    and `THIRD_PARTY_NOTICES.md` (DP3 citation).
- **Code:**
  - `src/roboforge/data/{pointcloud,pc_checks,pc_dataset}.py`
  - `src/roboforge/models/point_encoder.py`
  - `src/roboforge/train/pointcloud.py`
  - `src/roboforge/compute/{pc_train,launch_wait}.py`
  - `src/roboforge/eval/{pointcloud,pc_heldout}.py`
  - `scripts/{pc_feasibility,build_pc_dataset,benchmark_pc_training,pc_train_queue,eval_pc,run_pc_heldout,analyze_level2}.py`
  - `infra/modal_app.py`
  - `configs/models/pointcloud/diffusion_dp3_lift.yaml`
  - `configs/experiments/level2_pointcloud.yaml`
  - `configs/compute/modal.yaml`
  - `Makefile`
- **Tests:**
  - `tests/unit/test_{pointcloud,pc_train,level2_plan,launch_wait}.py`
  - `tests/simulator/test_{pointcloud_alignment,pc_closed_loop}.py`
  - `tests/research_integrity/test_pc_heldout_gate.py`
- **Data:** `data/manifests/pointcloud/lift_pc512_v1.json` (committed). The HDF5 file is
  gitignored.
- **Evidence:** `reports/session9/{pc_feasibility,pc_benchmark,pc_validation,pc_selection,level2_placement,pc_heldout,pc_heldout_rehearsal,level2_results}/`
  and `logs/session9/`.

## 4. Executed commands and results
| Command | Result |
|---|---|
| `make pc-feasibility` (container, `b16b692`) | PASS: interior depth error max 3.6 µm (0 pixels over 1 mm); known point within its footprint; invariants bitwise |
| `make pc-dataset` / `make pc-dataset-install` | 200 demos, 9,666 frames, content `39fa3e52…` |
| `scripts/pc_train_queue.py --budget-approved A-04` | 3/3 runs complete at 41.6–42.5 ms/step; 5 launches; $2.37 billed |
| `make pc-validate` × 3 runs, then `eval_pc.py select` | 36/36 clean; selected 30k, 40k, 60k (all 50/50) |
| `make pc-placement` × 12 | 12/12 clean. OOD: pc 32/28/25, rgb 50/49/48 of 50 |
| `make pc-heldout` × 3 (after the rehearsal) | 150/150 |
| `make level2-analyze` | H5 separated (−); results in `reports/session9/level2_results/` |
| `make lint`; full `pytest` alone; `make freeze-check`; both gates | clean; 478 passed; OK; open |

## 5. Key findings (validation numbers are development numbers)
- **H5 is separated (−):**
  - the point-cloud arm loses 43.3 points when the cube moves 5–7 cm outside the training
    square;
  - the RGB arm loses 2.0 points.
  - The point-cloud failures are mostly `no_grasp_near`: the gripper reaches the cube's
    region but misses the grasp.
- **In distribution the arms are equal:** 100 % on base and held-out, so Lift's ceiling holds
  (KI-034).
- **Interpretation limits:**
  - one encoder configuration, with no point-cloud augmentation, against an RGB arm trained
    with random crops;
  - world-frame coordinates are part of the treatment;
  - one task, 3 seeds;
  - not a DP3 reproduction.
- **Latency:** 146–147 ms per decision (point cloud) against 166–187 ms (RGB), on the
  container.

## 6. Decisions and issues
- **Decisions:** D-082–D-086 (D-086 amended in `08c3aa8`).
- **Issues:**
  - KI-045 (native CGL depth precision, mitigated);
  - KI-046 (hidden sites wrote depth, fixed);
  - KI-047 (client network loss stopped launches, mitigated by D-085);
  - KI-048 (the gate's Session 8 checks inside the container, fixed);
  - KI-049 (analysis training-row erratum, documented).

## 7. Budget
- **Session 9:** $2.37 billed (A-04, closed), against an estimate of about $3.2.
- **Project:** $10.61 billed in total. Modal month-to-date is $15.89 of $30.
- No approval is open. Evaluation was local, at $0.

## 8. Outstanding items (none blocks Session 10)
- Optional owner items: a Modal workspace budget; Docker cache pruning (KI-008); fixing the
  host's DNS (KI-047).
- Open from earlier sessions: KI-001 and KI-033 (EGL vs OSMesa), KI-034 (Lift ceiling).

## 9. What Session 10 can use
Frozen evidence for the Observatory:
- Level 1 (`reports/session7/`), the RGB study (`reports/session8/`) and Level 2
  (`reports/session9/`). Each has run manifests, checkpoint hashes and per-episode records
  under `runs/`.
- Calibrated point-cloud overlays (`reports/session9/pc_feasibility/…/lift_overlay_*.png`).
- The Level 2 results and erratum, for panel G "Beyond Pixels": Level 2 *was* executed, so
  real results may be shown, labelled DP3-inspired and negative for H5.

## 10. Exact starting instructions for Session 10
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 10 ONLY from MASTER_SPEC.md (RoboForge Observatory: interactive
3D scientific storytelling). Begin by reading all of MASTER_SPEC.md (especially §8 and the
Session 10 section), docs/STATUS.md, docs/HANDOFF.md "# Session 9 Handoff", docs/DECISIONS.md
(D-082–D-086), docs/KNOWN_ISSUES.md (KI-034, KI-045–KI-049), docs/EXPERIMENT_LEDGER.md
(Sessions 7–9 and the approvals table: no approval is open), docs/LEVEL1_COMPLETE.md,
docs/RGB_EXPERIMENT_FREEZE.md, docs/LEVEL2_EXPERIMENT_FREEZE.md, docs/LEVEL2_GO_NO_GO.md,
reports/session7/results/tables.md, reports/session8/rgb_results/tables.md and
reports/session9/level2_results/{tables.md,ERRATA.md}.

Inspect the real state first:
- git log --format=fuller -n 8; git status;
- make lint; make freeze-check;
- `uv run --frozen python scripts/run_rgb_heldout.py gate` and
  `uv run --frozen python scripts/run_pc_heldout.py gate` (both must print open);
- never run the full test suite alongside a container job (KI-043).

1. Record a decision for the Session 10 Makefile defaults (PREFIX ?= s10, evidence session 10;
   the D-082 pattern) and one for the web stack before adding any frontend dependency:
   - React Three Fiber / three.js per MASTER_SPEC §8.3;
   - a separate web/ package with its own lockfile;
   - uv.lock and pyproject unchanged.
2. Export authentic artifacts for the site from frozen evidence only:
   - recorded rollouts (link and object transforms re-derived from the saved simulator states on
     the canonical container), with model, seed, state and physics provenance;
   - actual diffusion denoising traces. If they need new policy calls, run them only on
     validation-regime starts (validation or the robustness sets), labelled as an export,
     never on held-out starts, and never as a new reported result.
   Verify the 3D viewer against at least 3 simulator reference frames and report every visual
   approximation.
3. Build the site sections of MASTER_SPEC §8.2 A–H:
   - findings with denominators and intervals from the frozen results files;
   - panel G shows the real Level 2 results (DP3-inspired, H5 separated (−), the KI-049
     erratum) and the calibrated overlays;
   - nothing illustrative is presented as a measured result.
4. Accessibility, keyboard, reduced motion, mobile layouts, loading states and asset budgets.
   `npm run build` (or equivalent) and E2E/UI sanity checks must pass. Hosting only if the
   owner authorizes it: ask, and record the answer.
5. STOP:
   - update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES and EXPERIMENT_LEDGER;
   - run lint, the full test suite (alone), freeze-check and both gates;
   - commit verified code as Zackary Holt <zackholt05@gmail.com>, with no AI co-author trailers;
   - inspect git log --format=fuller -n 5;
   - write the exact Session 11 prompt, then STOP.

Boundaries (binding):
- Level 1, the RGB study and Level 2 are frozen. Never rerun make eval-final, make
  analyze-final, make rgb-heldout, scripts/analyze_rgb_heldout.py, make pc-heldout, make
  pc-placement or make level2-analyze. Never edit reports/session6-9 results, the freezes,
  LEVEL1_COMPLETE or LEVEL2_GO_NO_GO's executed outcome. Never prune or move frozen runs.
- No training. No paid or remote compute (including hosting) without an approval row.
- No changes to data, eval suites, uv.lock or Python dependencies without a decision.
- Never uv sync or uv run inside the image.
- Use EVIDENCE_SESSION=10. Every rollout export runs on the canonical container via
  run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=10, from a clean tree.
- Label every estimate as an estimate.
```

# Session 8 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 8 (MASTER_SPEC §7.2 and Session 8: Level 1 RGB-conditioned policy study).
Status **PASS**, acceptance path (A).
- **Data:** camera-aligned RGB data verified: Lift, two cameras at 84².
- **Training:** visual BC-chunk and visual diffusion trained under a committed freeze
  (`d9c6217`): 6 runs on Modal L4 under A-03, **$8.06 billed**.
- **Selection:** both families validated (72 evaluations) and selected by §7.
- **Held-out test:** run once under a re-gated lock (D-081) and analysed once.
- **Level 1:** frozen and untouched.

**Result (Lift held-out, 3 seeds × 50 starts):**
- visual diffusion **100.0 %** [100.0, 100.0];
- visual BC-chunk **99.3 %** [96.0, 100.0];
- **diffusion − BC = +0.7 points [+0.0, +4.0], not separated**;
- engineering complete;
- one failure in total: BC s0 on start 49, `no_grasp_near`.

Lift saturates for every arm (KI-034).

## 2. Completed work
- **Prerequisites and KI-040** (smoke-sim built a held-out start): fixed.
- **Feasibility, no training** (E8-03–E8-05, D-074):
  - render path, alignment and projection tests;
  - a by-eye sample;
  - a clip with ground-truth actions;
  - cost at 84² and 128².
- **Gate D:** the A-02 L4 probe ($0.0317; D-076, D-077).
- **Gate D2:** "L60" (A-03, $15 cap; D-078).
- **Deterministic-CUDA amendment:** RESEARCH_PROTOCOL §13.
- **RGB dataset:** Lift 84², built on the container and installed with hash checks (D-079).
- **Freeze:** `docs/RGB_EXPERIMENT_FREEZE.md` (`d9c6217`, D-080).
- **Training** (E8-10): 6 runs × 60k steps in 12 launches.
  - Each run's first launch hit its 1 h deadline, then a resume finished it.
  - Modal preempted one resume (diffusion s2); its next attempt resumed from step 45,000
    (KI-044).
- **Validation and selection** (E8-11): 72/72 canonical. Every run reaches 50/50.
  - BC selects steps 5k, 5k and 55k.
  - Diffusion selects steps 55k, 55k and 60k.
- **Lock re-gate** (D-081, E8-12): `roboforge.eval.rgb_heldout`, with its test.
- **Rehearsal** (E8-13): bitwise equal to the validation episodes.
- **Held-out test** (E8-14): run once.
- **Frozen analysis** (E8-15): run once.
- **Failure videos** (E8-16).

## 3. Files created or changed (Session 8)
- **Code:**
  - `src/roboforge/compute/{rgb_bench,rgb_probe,rgb_train}.py`;
  - `src/roboforge/data/{rgb,rgb_dataset}.py`;
  - `src/roboforge/models/vision.py`;
  - `src/roboforge/train/rgb.py`;
  - `src/roboforge/eval/{rgb,rgb_heldout}.py`;
  - `infra/modal_app.py`: `rgb_probe`, `train_gpu`, `rgb-probe`, `train-gpu` and `fetch-rgb`.
- **Scripts:**
  - `benchmark_rgb_training`, `build_rgb_dataset`, `rgb_train_queue`, `eval_rgb`;
  - `run_rgb_heldout`, `analyze_rgb_heldout`, `render_rgb_episodes`.
  - The Makefile gains `rgb-dataset`, `rgb-dataset-install`, `rgb-validate`, `rgb-heldout`,
    `rgb-heldout-rehearse` and `rgb-heldout-gate`.
- **Configs:**
  - `configs/models/rgb/{bc_chunk,diffusion}_resnet18_lift.yaml`;
  - `configs/compute/modal.yaml`: `rgb_probe` and `train_gpu`.
- **Data:** `data/processed/robomimic_v1.5_ph/lift/rgb84_v1.hdf5`, with manifest
  `data/manifests/rgb/lift_rgb84_v1.json`.
- **Tests:**
  - `tests/unit/test_rgb_{probe,dataset,train,train_plan}.py`;
  - `tests/simulator/test_rgb_closed_loop.py`;
  - `tests/research_integrity/test_rgb_heldout_gate.py`.
- **Docs:**
  - `docs/RGB_EXPERIMENT_FREEZE.md`;
  - DECISIONS D-073–D-081;
  - KNOWN_ISSUES KI-040–KI-044, and the KI-034 note;
  - EXPERIMENT_LEDGER E8-01–E8-16, approvals A-02 and A-03;
  - COMPUTE_PLAN §6–§7, RESEARCH_PROTOCOL §13, STATUS and HANDOFF.
- **Evidence:**
  - `reports/session8/`: `rgb_feasibility`, `modal`, `rgb_benchmark`, `rgb_runs`,
    `rgb_validation`, `rgb_selection`, `rgb_heldout_rehearsal`, `rgb_heldout`,
    `rgb_results`, `rgb_videos`;
  - `logs/session8/`.
- **Runs** (local, uncommitted):
  - `runs/s8_rgb_*`: the 6 training runs, fetched and hash-verified;
  - `runs/rgb_eval/`;
  - `runs/rgb_heldout/`;
  - `runs/rgb_heldout_rehearsal/`.

  The Modal `roboforge-runs` Volume keeps every checkpoint and resume file.

## 4. Executed commands and results
- **Training queue:**
  `uv run --frozen python scripts/rgb_train_queue.py --budget-approved A-03`
  - from `d9c6217`, 04:25–13:18 UTC;
  - 6/6 complete, 0 failed launches, $8.06 billed.
- **Validation and selection:**
  - `make rgb-validate RGB_RUN=<run>`, 3 containers at a time;
  - then `uv run --frozen python scripts/eval_rgb.py select --run-id <run>`.
- **Gate:** `make rgb-heldout-gate` in the container: open, no problems.
- **Rehearsal:** `make rgb-heldout-rehearse RGB_RUN=<run> RGB_N=2` for diffusion s0 and BC s0.
- **Held-out test:** `make rgb-heldout RGB_RUN=<run>` × 6, from `a2c3c2e` clean, 13:56–14:04 UTC.
- **Analysis:** `ROBOFORGE_EVIDENCE_SESSION=8 uv run --frozen python scripts/analyze_rgb_heldout.py`,
  run once.
- **Videos:** `ROBOFORGE_EVIDENCE_SESSION=8 uv run --frozen python scripts/render_rgb_episodes.py`.
- **Checks:** `make lint`, `uv run --frozen pytest -q` (run alone; **427 passed**) and
  `make freeze-check` (OK).

## 5. Key findings (development and test numbers labelled)
- **Feasibility.** Lift demonstrations replay into exact-state RGB frames on the canonical
  container, and aligned frames can be produced cheaply. A ResNet-18 at 84² trains at
  59 ms/step (BC) and 78 ms/step (diffusion) on an L4 with deterministic kernels.
- **Validation (development numbers).** Lift is saturated for RGB:
  - every run reaches 50/50;
  - BC is at 50/50 from 5k steps;
  - diffusion climbs from 40–43/50 at 5k to 50/50 by 20–25k.
- **Held-out test.** Visual diffusion 100.0 % and visual BC-chunk 99.3 %; the difference is
  not separated.
  - Visual control from two 84² cameras plus 9-D proprioception sits at the same Lift ceiling
    as the Level 1 low-dim policies (BC-MLP-chunk 98.0 %, diffusion 97.3 %). This is context
    only, not a declared comparison.
  - Lift cannot rank visual BC against visual diffusion.
- **Cost and latency.**
  - Diffusion costs 1.35× the training (1.29–1.36 h, $1.52–1.60 per run, against BC's
    1.00–1.01 h and $1.13–1.14).
  - Diffusion's per-decision inference is 2.7× BC's: 166–187 ms against 63–66 ms (CPU, one
    thread).
- **The three kinds of negative result stay apart:**
  - engineering: none, and engineering is complete;
  - validation: no family failed it;
  - held-out: the difference is not separated.

## 6. Decisions and issues
- **Decisions:** D-073–D-081. D-081 re-gates the held-out lock on the RGB freeze, and fixes
  the analysis choices the freeze left open.
- **Issues:**
  - KI-040 (fixed);
  - KI-041 (CGL first render, mitigated);
  - KI-042 (the frozen diffusion sampler is CPU-only; open, low);
  - KI-043: host memory stall. **Never run the full suite alongside a container job.**
  - KI-044 (Modal preemption, mitigated);
  - KI-034 now also covers RGB saturation.

## 7. Budget
| Approval | Status | Spend |
|---|---|---|
| A-01 (Session 5) | closed | $0.148 |
| A-02 (Gate D probe) | closed | $0.0317 |
| A-03 (Gate D2, 6 RGB runs) | closed | $8.06 |

- Project total billed: **$8.24**.
- Month-to-date Modal usage was $5.45 of the $30 credit before A-03. Recheck it with
  `/opt/anaconda3/bin/python -m modal billing report` before proposing a Session 9 cap.
- **No approval is open.**

## 8. Outstanding items (none blocks Session 9)
- RGB on Can: about 100k diffusion steps, about $15 (COMPUTE_PLAN §7.4, an estimate). Not
  approved and not planned.
- KI-042: CUDA inference for the frozen sampler. Not needed; inference stays on the CPU.
- Optional: a Modal workspace budget, and pruning the Docker cache (KI-008).

## 9. Owner intent for Session 9 (recorded 2026-11-09, 07:57 EST)
The owner said: "session 9 is a go, we're doing it all today".
- Session 9 (Level 2, DP3-inspired point clouds) follows the GO path.
- It still opens with `docs/LEVEL2_GO_NO_GO.md`, listing the blockers, as MASTER_SPEC
  requires.
- It needs a **new capped approval row** before any GPU training, because A-03 covered only
  the 6 RGB runs.
- **Proposed scope (an estimate, to be re-measured):**
  - Lift only, with point clouds from the same two cameras' depth;
  - DP3-inspired diffusion against the frozen RGB diffusion arm;
  - 3 seeds × 60k steps on the L4: about 4 h of GPU and about $5 at list price.
- **Design point for the GO document.** RGB diffusion already scores 100 % on Lift held-out.
  - A held-out comparison on in-distribution Lift can therefore only come out "not
    separated" or show a point-cloud deficit.
  - MASTER_SPEC §7.3 asks for in-distribution plus precommitted object-placement changes.
    The committed Lift `ood_cube_shift` set (D-063) is a candidate.
  - Evaluating the frozen RGB diffusion checkpoints on it would be a new, preregistered
    validation-regime evaluation, never a held-out rerun.
  - This is for the owner to decide in the GO document.

## 10. Exact starting instructions for Session 9
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 9 ONLY from MASTER_SPEC.md (Level 2: explicit 3D perception,
DP3-inspired, optional conditional experiment). Begin by reading all of MASTER_SPEC.md
(especially §7.3 and the Session 9 section), docs/STATUS.md, docs/HANDOFF.md "# Session 8
Handoff" (§9 records the owner's GO intent), docs/DECISIONS.md (D-073–D-081),
docs/KNOWN_ISSUES.md (KI-034, KI-040–KI-044), docs/EXPERIMENT_LEDGER.md (Session 8 and the
approvals table: no approval is open), docs/COMPUTE_PLAN.md §6–§7,
docs/RGB_EXPERIMENT_FREEZE.md, docs/LEVEL1_COMPLETE.md and RESEARCH_PROTOCOL §7, §9, §10, §13
and §14.

Inspect the real state first:
- git log --format=fuller -n 8; git status;
- make lint; make freeze-check;
- `uv run --frozen python scripts/run_rgb_heldout.py gate` (it must print open);
- never run the full test suite alongside a container job (KI-043).

1. Write docs/LEVEL2_GO_NO_GO.md, covering (a)–(f) of MASTER_SPEC §7.3:
   - the hypothesis and baseline: the matched reference is the frozen RGB diffusion arm,
     whose held-out results are reused as committed and never rerun;
   - the compute and dataset budget;
   - reproducible depth/point-cloud extraction from the demonstration states;
   - verified camera projection and frames;
   - the comparison protocol;
   - an explicit fallback.
   Enumerate the technical, licensing and resource blockers. Address Lift's ceiling: RGB
   diffusion is at 100 % held-out. Propose whether to add a preregistered object-placement
   evaluation, e.g. on the committed Lift ood_cube_shift starts, for both arms.
2. Feasibility, with no training:
   - Render depth from the same two cameras through the D-074 path.
   - Convert it to metric depth (near/far and the depth-buffer convention).
   - Back-project to XYZ with logged intrinsics and extrinsics.
   - Add a numerical test that projects a known simulator point to pixels and back within
     tolerance, and an overlay image of the points on the robot geometry.
   - Define the workspace crop and a reproducible N-point sampling.
   - State whether any segmentation or simulator label is used: sensor-like depth plus
     calibration only, otherwise label the run "privileged geometry".
   - Measure extraction and training throughput, and the point-cloud dataset size.
3. Present the GO/NO-GO and a budget proposal with measured numbers, labelled as estimates.
   Then STOP and ask the owner. Record the answer as a new approval row (A-04) before any
   paid launch.
4. If GO: write docs/LEVEL2_EXPERIMENT_FREEZE.md before the first training run.
   - The arm is a DP3-inspired point encoder plus the same diffusion head as the RGB arm.
   - Match the RGB arm on demonstrations, proprioception, held-out starts, action
     normalization, horizons, optimizer, budget, selection rule and sampling rate.
   - Then train, validate and select under it.
   - Before any Level 2 held-out rollout, re-gate the held-out lock with a recorded decision
     and a test, in the D-081 pattern: a new module, no frozen file changed, guarded tests.
   - Run the held-out test once and analyse it once with the §9 bootstrap. Label it
     DP3-inspired.
   If NO-GO: document the skip and the reasons in LEVEL2_GO_NO_GO.md, and do not train.
5. STOP:
   - Update STATUS, HANDOFF, DECISIONS, KNOWN_ISSUES, EXPERIMENT_LEDGER and COMPUTE_PLAN.
   - Run lint, the full test suite (alone) and freeze-check.
   - Commit verified code as Zackary Holt <zackholt05@gmail.com>, with no AI co-author
     trailers.
   - Inspect git log --format=fuller -n 5.
   - Write the exact Session 10 prompt, then STOP.

Boundaries:
- Level 1 and the RGB study are frozen. Never rerun make eval-final, make analyze-final,
  make rgb-heldout or scripts/analyze_rgb_heldout.py. Never edit reports/session6-8 results,
  the freezes or LEVEL1_COMPLETE. Never prune or move frozen runs.
- No paid or remote compute without an approval row.
- No changes to data, eval suites, uv.lock or dependencies without a decision.
- Never uv sync or uv run inside the image.
- Launch reported runs from a clean tree, and use TRAIN_RUN, not RUN.
- Use EVIDENCE_SESSION=9 and PREFIX=s9. Every reported, selection or test rollout runs on the
  canonical container via run_sim_cpu.sh with ROBOFORGE_EVIDENCE_SESSION=9.
- Label every estimate as an estimate.
```

# Session 7 Handoff

Do not trust this file blindly: re-run the checks in "Exact starting instructions".

## 1. Session completed
Session 7 (MASTER_SPEC: Level 1 Can, paired held-out testing, robustness, scientific freeze).
Status **PASS**. Level 1 is complete and frozen in `docs/LEVEL1_COMPLETE.md`.
- **Can main suite:** 24 runs and 480 validation evaluations.
- **Robustness and ablations:** H4 plus A1–A3 on preregistered fresh starts, 135
  evaluations, validation regime.
- **Freeze:** committed before any held-out rollout (`bf0a0fa`).
- **Held-out test:** run once, 26 evaluations, then analysed once (`97cc48a`).

All work was local at $0. No run failed.

## 2. Completed work
- **Can main suite** (E7-02–E7-04):
  - Started from `96aa972`. It crashed at 02:14 UTC (KI-036) and was fixed in D-070
    (`f0715c4`), then restarted.
  - 24/24 runs complete, 480/480 evaluations, failed none.
  - Chosen trials: BC-MLP 2, BC-RNN 2, BC-MLP-chunk 4 and diffusion 2.
- **Can latency, figures and test configuration** (E7-06, E7-07):
  - `reports/session7/cost_latency/`, `figures/` and `can_test_configuration.json`.
- **Preregistration, RESEARCH_PROTOCOL §14** (`2859d04`):
  - the robustness starts (`5d9b8c0`; container check bitwise after KI-037, `005bc94`);
  - the failure taxonomy, D-065;
  - the bootstrap, D-066;
  - the held-out path with its rehearsal, D-067 and D-069.
- **Robustness (H4) and ablations A1, A2 and A3** (E7-09–E7-11):
  - Evaluated from `005bc94` and analysed from `565c6e4` after KI-038.
- **Rehearsal** of the held-out path on the validation suite (E7-12): PASS.
- **Freeze** (E7-13, D-072, KI-039):
  - Container attestation: locked before the freeze, unlocked after it.
  - The freeze is identical to the rehearsed draft except for its date.
- **Held-out test** once, then the frozen analysis once (E7-14).
- **Records:** `docs/LEVEL1_COMPLETE.md`, plus all the STOP docs.

## 3. Files created or changed (Session 7)
- **Code:**
  - Can plan: `configs/experiments/level1_core_can.yaml`.
  - A3 plan and config: `ablation_a3_can.yaml`, `configs/models/diffusion_unet_to1_can.yaml`.
  - `src/roboforge/sim/robustness.py`, `configs/eval/{robustness,failure_taxonomy}.yaml`.
  - `src/roboforge/eval/{taxonomy,statistics,freeze,heldout}.py` and inference variants.
  - Scripts: `generate_robustness_states`, `run_evals`, `analyze_robustness`,
    `failure_taxonomy`, `write_freeze_block`, `run_heldout`, `eval_heldout`,
    `analyze_heldout`, `check_rehearsal`.
  - Changes: the D-070 guards in `run_suite.py`; the host-wide container slots
    (`roboforge.slots`).
  - Tests for each of these.
- **Data:** `data/eval_states/<task>/{robustness_base,ood_*}`, 5 sets × 50 (D-063).
- **Docs:**
  - `docs/FINAL_TEST_FREEZE.md` and `docs/LEVEL1_COMPLETE.md`;
  - RESEARCH_PROTOCOL §14;
  - DECISIONS D-060–D-072;
  - KNOWN_ISSUES KI-036–KI-039;
  - EXPERIMENT_LEDGER E7-01–E7-15, with cost and tuning accounting;
  - COMPUTE_PLAN, STATUS and HANDOFF.
- **Evidence:**
  - `reports/session7/`: suite, validation, selection, tuning, cost_latency, figures,
    robustness, heldout and results.
  - `logs/session7/`: suite, robustness, rehearsal, heldout, analysis and `WORKLOG.md`.
  - Checkpoints and episode records stay in `runs/` (gitignored).

## 4. Executed commands and results
See EXPERIMENT_LEDGER E7-01–E7-15. Every reported rollout ran on the canonical container
through `infra/containers/run_sim_cpu.sh` from a clean commit:
- **Training:** 27 runs, all `complete`.
- **Evaluations:** 667 in all, with 0 errors and every audit passing.
  - 480 Can validation;
  - 135 robustness and ablation;
  - 26 rehearsal;
  - 26 held-out.
- **Final checks:** `make lint` is clean and `make test` gives **372 passed** at `bf0a0fa`.
  They were re-run at the STOP commit (E7-15).

## 5. Key findings: held-out test, frozen protocol (`docs/LEVEL1_COMPLETE.md` §2–§3)
- **Success, % of 3 seeds × 50 held-out starts:**

  | Family | Lift | Can |
  |---|---|---|
  | BC-MLP | 90.7 | 76.7 |
  | BC-RNN | 100.0 | 96.0 |
  | BC-MLP-chunk | 98.0 | 96.0 |
  | Diffusion | 97.3 | 95.3 |
  | Random | 0 | 0 |

- **H1:** separated for every family on both tasks.
- **H2, diffusion − BC-MLP:** on Can +18.7 pp [+4.7, +32.7], separated. On Lift +6.7 pp,
  not separated in success, but separated in the secondary episode length.
- **H2, diffusion − BC-RNN:** not separated on either task.
- **H2-c, diffusion − BC-MLP-chunk:** not separated on either task, so the data do not
  attribute any diffusion advantage to the generative model.
- **H3, BC-RNN − BC-MLP:** separated on both tasks, +9.3 and +19.3 pp. A3 (T_o within
  diffusion) is not separated.
- **H4:** every family drops under every OOD set.
  - Diffusion degrades least on the Lift arm offset: 88.0 %, against 22.7–50.0 % for the BC
    families.
  - On the Can arm offset, diffusion (71.3 %) and BC-MLP-chunk (79.3 %) are not separated.
- **A1 (EMA) and A2 (DDIM-10 vs DDPM-100):** not separated on either task.
- **Selection optimism:** validation scores exceed fresh-start and held-out scores. On Can the
  gap is up to 13.4 pp on fresh starts and up to 6.0 pp on the held-out starts (BC-MLP;
  descriptive).

## 6. Open decisions (Session 8)
- **Gate D (RGB visual policies):** the owner must approve a budget before any RGB main
  training, from a measured pilot.
  - Local CPU only at $0, or a capped Modal approval row.
- **Resolution 84×84 or 128×128:** from the alignment and throughput pilot (MASTER_SPEC §7.2).
- **Encoder regime:** from scratch, pretrained and frozen, or fine-tuned. Shared across
  visual BC and visual diffusion.
- **Disk:** RGB datasets may not fit in the 22 GB free. Pruning the Docker build cache
  (KI-008) is an owner decision.

## 7. Decisions
D-060–D-072 (Session 7). D-072 records how the freeze and the held-out run were executed.

## 8. Outstanding issues (none blocks Session 8)
- **KI-001 / KI-033:** EGL vs OSMesa on Modal is still unsettled. OSMesa works in the
  canonical container; check it for RGB throughput.
- **KI-008:** disk and the Docker build cache.
- **KI-034:** Lift is near ceiling.
- **KI-035:** a shared host inflates wall times.
- **KI-028:** R = 1.
- All Session 7 issues (KI-036–KI-039) are FIXED.

## 9. Budget
- **Session 7:** $0. 46.19 CPU core-h of training and 64.25 container-h of evaluation.
- **Level 1 in all:** $0.148, from Session 5 Modal Gate A.
- No approval row is open. Any Modal or GPU use needs a new row with a cap first.

## 10. Exact starting instructions for Session 8
Paste into a fresh Claude Code session:

```text
Implement RoboForge 3D SESSION 8 ONLY from MASTER_SPEC.md (Level 1 RGB-conditioned policy,
separate feasibility/budget gate; MASTER_SPEC §7.2 and Session 8). Begin by reading all of
MASTER_SPEC.md, docs/STATUS.md, docs/HANDOFF.md, docs/LEVEL1_COMPLETE.md,
docs/FINAL_TEST_FREEZE.md, docs/DECISIONS.md (through D-072), docs/KNOWN_ISSUES.md
(especially KI-001, KI-008, KI-018, KI-019, KI-020, KI-022, KI-033, KI-034, KI-035, KI-039),
docs/EXPERIMENT_LEDGER.md (Sessions 6-7), docs/RESEARCH_PROTOCOL.md (all amendments
binding, §14 included), docs/SIMULATOR_CONTRACT.md, docs/DATA_DICTIONARY.md and
docs/COMPUTE_PLAN.md, and inspect the actual repository/test state. Do not blindly trust
prior handoff claims: first run `make setup`, `make download`, `make check-data`,
`make build-data`, `make check-eval-states`, confirm `git status --short data/` is empty,
then `make lint`, `make test`, `make freeze-check`,
`make smoke-sim SMOKE_OUT=logs/session8/prereq`, and on the canonical container
`ROBOFORGE_EVIDENCE_SESSION=8 make in-sim CMD="python scripts/smoke_sim.py --task lift --out-dir logs/session8/prereq/container"`
and the same with `--task can`; confirm
`docker image inspect -f '{{.Id}}' roboforge-sim-cpu:session0` equals the image_id in
configs/eval/platform.yaml; confirm the Level 1 freeze is intact:
`git diff 97cc48a -- docs/FINAL_TEST_FREEZE.md reports/session6/lift_test_configuration.json reports/session7/can_test_configuration.json reports/session7/heldout reports/session7/results`
is empty and `git log --oneline -- docs/LEVEL1_COMPLETE.md` lists exactly one commit (the
Session 7 STOP commit; report, do not repair, if not); run git log --format=fuller -n 5 to confirm the
verified commits and the identity Zackary Holt <zackholt05@gmail.com>. Pass
EVIDENCE_SESSION=8 and PREFIX=s8 (or bump the Makefile defaults as a recorded change) so
Session 8 never writes into Session 6-7 evidence.

HARD SCIENTIFIC BOUNDARY (Session 7 STOP): Level 1 state-based results are frozen. Never
rerun make eval-final or make analyze-final, never re-evaluate any Level 1 configuration on
the held-out suites, never edit reports/session6-7 evidence, the freeze or LEVEL1_COMPLETE,
and never let RGB results modify core selection, labels or findings. RGB results are a
separately labelled observation-modality study (MASTER_SPEC §3.2 item 5); state-based Level 1
is an oracle-style comparator because it sees privileged object state.
The held-out lock is now OPEN for every caller (the Level 1 freeze is committed, so
eval_states.lock_status returns unlocked): that is NOT permission. No Session 8 code may
load, render, reset to or otherwise read the final_test suite or its seeds for any purpose
until docs/RGB_EXPERIMENT_FREEZE.md is committed; before any RGB held-out work, re-gate the
lock on that file (or on a new, separately locked RGB suite) with a test, as a recorded
decision, and decide in the RGB freeze whether the RGB test reuses the Level 1 held-out
states or a new suite. Never prune or move the 24 frozen run directories, the A3 runs or
runs/*__heldout* (gitignored; the only copies behind the freeze's checkpoint hashes and
the results' taxonomy), even if disk runs short.

Session 8 goal (MASTER_SPEC Session 8), in this order:
(1) Feasibility, no training yet: from data/raw/robomimic_v1.5/{lift,can}/ph/demo_v15.hdf5
(per-demo `states` and `model_file`; no images in the file), restore raw simulator states
through RoboForgeEnv-compatible code (never mj_forward on live evaluation data, KI-019;
never robosuite transform helpers for precision-critical values, KI-018), render RGB
headless with the exact camera name(s) and backend documented (OSMesa in the canonical
container; EGL on Modal is still unsettled, KI-001/KI-033), and verify exact time-step
alignment (task, demo/episode id, step_index, state hash, camera intrinsics/extrinsics,
model XML hash, frame hash). Create a small demonstrably aligned clip with ground-truth
actions, inspect a fixed random sample by eye, and add a code test comparing the projected
object position with its visible location (segmentation where available). Measure render
throughput and storage for 84x84 and 128x128 on this host. If raw data or rendering fails,
mark the RGB study NOT EXECUTED with details and reproducible evidence, publish that
feasibility note honestly, update the docs and STOP (no fake or untrained "visual control").
(2) Gate D budget proposal and owner approval: from the measured pilot, propose seeds x
steps x encoder/resolution x evaluation cost (local CPU at $0, or an exact capped Modal plan
with upper bounds), record it in docs/COMPUTE_PLAN.md and STOP to ask the owner; record the
owner's decision as an approval row in docs/EXPERIMENT_LEDGER.md before any main RGB
training. No paid or remote compute without that row.
(3) RGB experiment freeze, committed before the first RGB training run: a fair visual BC vs
visual diffusion comparison on Lift (optionally Can), RGB-only meaning robot proprioception
+ camera image and NO object ground-truth state (check that object_state never leaks
through normalisation or features), declared in ObservationSpec and every run manifest;
one shared encoder architecture and initialization (e.g. a compact ResNet), identical
augmentations, preprocessing, proprioceptive features, camera settings, budget, selection
rule (§7 on validation starts only) and evaluation states; the held-out RGB test initial
states, run count, bootstrap and reporting rules, written into a new docs/RGB_EXPERIMENT_FREEZE.md
(name it; do not edit docs/FINAL_TEST_FREEZE.md) before any RGB held-out rollout.
(4) Train and evaluate under that freeze and the approved budget; produce real
RGB-conditioned closed-loop results on the canonical container and representative failure
videos; if unsuccessful, publish that fact. Label every RGB number by modality, data and
compute budget, and compare visual-only to visual-only for architectural claims.

Compute: Gate C covered the Level 1 main suite (local CPU only, $0; 1 torch thread per run,
at most 8 concurrent training runs, at most 4 evaluation containers in parallel); Session 8
needs its own Gate D decision before main RGB training, and any Modal use needs a new owner
approval row first. Disk is tight (22 GB free at the end of Session 7; KI-008): measure RGB
storage before generating datasets. Track failed runs too. Measure latency on the canonical
container with nothing else running.

Keep the Session 1-7 contracts (DECISIONS D-010, D-017..D-072): training data only through
committed, hashed, versioned datasets (extend SequenceDataset.from_committed or add a
recorded RGB equivalent); policy observations only from RoboForgeEnv; outputs decoded with
ActionCodec; learned policies implement roboforge.eval.policies.Policy and run with
run_episode + verify_record; stochastic policies seeded by the protocol's noise-seed rule;
resets only from stored snapshots; first success or 400 steps; model and checkpoint
selection only on validation starts; held-out states read only through
src/roboforge/sim/eval_states.py and only under a committed freeze; every reported,
selection or test rollout on the canonical container via infra/containers/run_sim_cpu.sh
with ROBOFORGE_EVIDENCE_SESSION=8 (never uv sync/uv run inside the image, KI-020). Native
macOS rollouts are development-only; checkpoints and anything a container rollout reads
live under runs/<run_id>/; every run writes an immutable manifest with git SHA + dirty flag
(launch reported runs from a clean committed tree; do not edit tracked files or commit
while they run), environment, platform, data/spec hashes, full config, seeds, parameter
count, checkpoint hashes and wall/CPU time; compare checkpoints by content, not file hash
(KI-029); report motion-dimension clipping separately from the gripper (KI-022); offline
validation MSE never selects (KI-024). Use the Make variable TRAIN_RUN (not RUN) for run
directories. No paid or remote compute and no remote publication without explicit owner
approval; do not change committed data artifacts, eval suites, uv.lock or dependencies
without a recorded decision (an image/rendering dependency needs one).

Use the current verified simulator/data/checkpoint/manifest contracts. Run relevant tests
and record commands, real results and provenance. Never leak held-out test states into model
selection; never invent results or costs (label estimates as estimates). Use Zackary Holt
as the sole named RoboForge project author with the verified local Git identity; no AI
co-author trailers, while preserving external scientific and license credit (Diffusion
Policy, robomimic, diffusers, robosuite, MuJoCo and any vision backbone used). At the
SESSION 8 STOP boundary (core Level 1 metrics remain frozen and unchanged), update
docs/STATUS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/KNOWN_ISSUES.md,
docs/EXPERIMENT_LEDGER.md (all runs, including failed ones, and the all-in cost record) and
docs/COMPUTE_PLAN.md; commit verified code only, inspect commit metadata
(git log --format=fuller -n 5), write the exact prompt for SESSION 9, and STOP. Do NOT
proceed automatically.
```
