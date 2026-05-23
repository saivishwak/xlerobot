/** Thin API client. SWR is used in components for cache + revalidation. */

export type CheckStatus = "ok" | "warn" | "fail" | "info";
export interface DoctorCheck { name: string; status: CheckStatus; detail: string }

export interface CameraSpec {
  name: string;
  path: string;
  width: number; height: number; fps: number; fourcc: string;
  role: string | null;
  by_path: string | null;
  card: string;
}

export type ArmSide = "left" | "right";

export interface VRControllerPose {
  /** present, in metres (lerobot-frame XYZ) */
  position: [number, number, number] | null;
  /** quaternion x,y,z,w from VR — null if no goal received yet */
  rotation: [number, number, number, number] | null;
  trigger: boolean;
  thumbstick: { x: number; y: number } | null;
  /** ms since last goal arrived from the controller */
  age_ms: number | null;
  /** "idle" | "position" | "reset" */
  mode: string;
}

/** Per-arm state in the new bimanual view. */
export interface VRArmState {
  connected: boolean;
  /** When false, motors are limp — user is hand-posing this arm. Drive loop
   *  skips arms with torque off. */
  torque_enabled: boolean;
  calibrated: boolean;
  joint_target: Record<string, number>;
  controller: VRControllerPose;
  /** Calibration diagnostics — populated after the user squeezes grip on this
   *  arm's controller. Shows what the backend thinks the mapping is doing. */
  calibration: {
    /** Robot EE position (m, robot base frame) at the moment of RESET. */
    anchor_ee_pos: [number, number, number];
    /** Cumulative offset from anchor in robot base frame (m, unclamped). */
    offset_robot: [number, number, number];
    /** Final EE target (m, robot base frame) — anchor + offset, clamped. */
    target_ee_pos: [number, number, number];
    /** Yaw (deg) of the active session VR→robot frame relative to default. */
    session_yaw_deg: number;
    /** Guided 3-vector calibration wizard state. */
    wizard_state:
      | "idle"
      | "awaiting_anchor_fwd"  | "motioning_fwd"
      | "awaiting_anchor_up"   | "motioning_up"
      | "awaiting_anchor_left" | "motioning_left";
    /** Live motion magnitude (m) accumulated during the current motion-capture. */
    wizard_motion_m: number;
    wizard_target_m: number;
    wizard_min_m: number;
    /** Most-recently-captured forward / up / left motion magnitudes (m). */
    wizard_last_fwd_m: number;
    wizard_last_up_m: number;
    wizard_last_left_m: number;
    wizard_fwd_captured: boolean;
    wizard_up_captured: boolean;
    wizard_left_captured: boolean;
    /** Result of the lateral-check step. True = matrix mirroring detected,
     *  invert_lateral was auto-flipped on by the wizard. */
    invert_lateral: boolean;
    /** "good" = captured motion vectors well-separated, matrix is robust.
     *  "poor" = vectors too parallel (cos > 0.6); re-run wizard for better
     *  results. */
    confidence: "good" | "poor";
    /** Whether the most-recent calibration has been written to
     *  `config/vr_calibration.yaml` and reload on next startup. */
    persisted: {
      saved: boolean;
      calibrated_at: string | null;
      forward_motion_m: number;
      up_motion_m: number;
    };
  };
  /** Per-arm home pose state — read from config/xlerobot.yaml + live flag. */
  home: {
    captured: boolean;
    joints: Record<string, number>;
    homing: boolean;
  };
}

export interface VRStatus {
  /** New: per-arm state, keyed by side. */
  arms: { left: VRArmState; right: VRArmState };
  /** Sides that currently have a motor connection. */
  connected_sides: ArmSide[];
  /** Which arm VR is currently driving (engage-gated bimanual). null = none. */
  active_arm: ArmSide | null;
  /** Global engage gate. To actually move motors: engaged && active_arm != null && that arm calibrated */
  engaged: boolean;
  /** Whether dataset recording is currently active (toggled by B button or UI). */
  recording: boolean;
  /** Detail on the LeRobotDataset state. Useful for the UI's Recording card. */
  recording_info: {
    active: boolean;
    episodes_saved: number;
    frames_in_current_episode: number;
    repo_id: string | null;
    /** Most-recent task description (from UI or previous session). */
    last_task: string;
    /** Absolute filesystem path where datasets are/will be written. */
    root: string;
  };
  /** 0.1..1.0 — multiplier on VR delta caps */
  scale: number;
  /** ms since the last drive-loop tick */
  last_tick_age_ms: number | null;
  /** session-level error message, set after a failure (connect, send, etc.) */
  last_error: string | null;
  /** present joints for ALL connected arms (prefixed keys) */
  joint_present: Record<string, number>;
  /** calibration bounds in degrees, per joint */
  joint_bounds: Record<string, [number, number]>;
  /** URL the user should open on the Quest browser */
  vr_endpoint: string | null;

  // ─── Legacy fields (deprecated) — mirror the active or first-connected arm.
  // Kept so the existing single-arm UI keeps working until it migrates to `arms`.
  /** @deprecated use `active_arm` or `arms` */
  arm: ArmSide | null;
  /** @deprecated use `connected_sides.length > 0` or `arms[side].connected` */
  connected: boolean;
  /** @deprecated use `arms[active_arm].calibrated` */
  vr_calibrated: boolean;
  /** @deprecated use `arms[active_arm].joint_target` */
  joint_target: Record<string, number>;
  /** @deprecated use `arms[active_arm].controller` */
  controller: VRControllerPose;
  /** @deprecated use `arms[active_arm].controller.age_ms` */
  last_goal_age_ms: number | null;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { "content-type": "application/json" }, ...init });
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status} ${r.statusText}: ${text}`);
  }
  return r.json();
}

export const fetcher = <T,>(url: string) => req<T>(url);

export const api = {
  doctor: () => req<{ checks: DoctorCheck[] }>("/api/doctor"),
  cameras: () => req<{ cameras: CameraSpec[]; roles: string[] }>("/api/cameras"),
  assign:  (by_path: string, role: string | null) =>
    req<{ ok: true; cameras: CameraSpec[] }>("/api/cameras/assign", {
      method: "POST", body: JSON.stringify({ by_path, role }),
    }),

  vrStatus: () => req<VRStatus>("/api/vr/status"),
  vrConnect: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/connect", { method: "POST", body: JSON.stringify({ arm }) }),
  /** Pass `arm` to disconnect one side; omit to disconnect both. */
  vrDisconnect: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/disconnect", {
      method: "POST",
      body: JSON.stringify(arm ? { arm } : {}),
    }),
  /** When both arms are connected, `active_arm` must be set to choose which one VR drives. */
  vrEngage: (engaged: boolean, scale: number, active_arm?: ArmSide) =>
    req<VRStatus>("/api/vr/engage", {
      method: "POST",
      body: JSON.stringify({ engaged, scale, ...(active_arm ? { active_arm } : {}) }),
    }),
  vrEmergencyStop: () =>
    req<VRStatus>("/api/vr/emergency_stop", { method: "POST" }),
  /** UI-side mirror of the B button on the right Quest controller. Pass the
   *  per-episode task description; LeRobot v2 stores it on every frame and
   *  uses it as conditioning input for VLA training. The optional `root` arg
   *  overrides where the dataset is written (null/empty = HF default). */
  vrSetRecording: (enabled: boolean, task: string = "", root: string = "") =>
    req<VRStatus>("/api/vr/recording", {
      method: "POST", body: JSON.stringify({ enabled, task, root }),
    }),
  /** Begin a guided motion-based calibration for one arm. */
  vrCalibrateStart: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/start", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  vrCalibrateCancel: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/calibrate/cancel", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  /** Read present joints for one arm (or all connected if arm omitted) and
   *  write to config/xlerobot.yaml's home_pose block. */
  vrHomeCapture: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/home/capture", {
      method: "POST", body: JSON.stringify(arm ? { arm } : {}),
    }),
  vrHomeGo: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/home/go", {
      method: "POST", body: JSON.stringify(arm ? { arm } : {}),
    }),
  vrHomeCancel: (arm?: ArmSide) =>
    req<VRStatus>("/api/vr/home/cancel", {
      method: "POST", body: JSON.stringify(arm ? { arm } : {}),
    }),
  /** Disable torque on one arm so the user can hand-pose it. */
  vrTorqueRelease: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/torque/release", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
  /** Re-enable torque on one arm at its CURRENT position (no snap-back). */
  vrTorqueLock: (arm: ArmSide) =>
    req<VRStatus>("/api/vr/torque/lock", {
      method: "POST", body: JSON.stringify({ arm }),
    }),
};
