import {
  Alert, Badge, Box, Button, Card, CopyButton, Container, Grid, Group, Modal,
  SegmentedControl, SimpleGrid, Slider, Stack, Switch,
  Text, TextInput, Title, Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import {
  IconAlertTriangleFilled, IconCopy, IconCheck, IconHandStop,
  IconHeadset, IconHome, IconHomeCog, IconLock, IconLockOpen,
  IconPlayerPlay, IconPlayerStop,
  IconPlugConnected, IconPlugConnectedX,
} from "@tabler/icons-react";
import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { ArmSide, VRArmState, VRStatus, api, fetcher } from "../api";

const JOINTS_ORDER = [
  "shoulder_pan", "shoulder_lift", "elbow_flex",
  "wrist_flex", "wrist_roll", "gripper",
] as const;

const NEAR_BOUND_DEG = 5;

function fmt(n: number | undefined, digits = 1): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function ageBadge(ms: number | null | undefined, prefix = "") {
  if (ms === null || ms === undefined) {
    return <Badge size="xs" variant="light" color="gray">{prefix}—</Badge>;
  }
  const color = ms < 200 ? "teal" : ms < 500 ? "yellow" : "red";
  return <Badge size="xs" variant="light" color={color}>{prefix}{ms} ms</Badge>;
}

/** Per-arm Connect / Disconnect row. Both arms are always shown so the user can
 *  bring up either one independently — bimanual is "both can be torqued, only one
 *  drives at a time" (see VRTeleopSession._active_arm). */
function ConnectionBar({ s, onConnect, onDisconnect, onEStop, busy }: {
  s: VRStatus | undefined;
  onConnect: (arm: ArmSide) => void;
  onDisconnect: (arm: ArmSide) => void;
  onEStop: () => void;
  busy: boolean;
}) {
  const armCard = (side: ArmSide) => {
    const armState = s?.arms?.[side];
    const connected = !!armState?.connected;
    const isActive = s?.active_arm === side;
    return (
      <Card withBorder padding="sm" key={side}
            style={{ borderColor: isActive ? "var(--mantine-color-red-7)" : undefined }}>
        <Group justify="space-between" wrap="nowrap">
          <Group gap="xs">
            <Text fw={600} fz="sm" tt="capitalize">{side} arm</Text>
            {connected ? (
              <Badge color={isActive ? "red" : "teal"} variant="filled" size="sm">
                {isActive ? "ACTIVE" : "ready"}
              </Badge>
            ) : (
              <Badge color="gray" variant="light" size="sm">offline</Badge>
            )}
            {connected && (
              <Badge color={armState?.calibrated ? "teal" : "gray"}
                     variant="light" size="sm">
                {armState?.calibrated ? "anchored" : "press grip to anchor"}
              </Badge>
            )}
          </Group>
          {!connected ? (
            <Button size="xs" leftSection={<IconPlugConnected size={14} />}
                    onClick={() => onConnect(side)} loading={busy}>
              Connect
            </Button>
          ) : (
            <Button size="xs" color="gray" variant="default"
                    leftSection={<IconPlugConnectedX size={14} />}
                    onClick={() => onDisconnect(side)} loading={busy}>
              Disconnect
            </Button>
          )}
        </Group>
      </Card>
    );
  };

  return (
    <Card withBorder padding="md">
      <Group justify="space-between" wrap="nowrap" mb="sm">
        <Text fw={700} fz="md">Arms</Text>
        <Tooltip label="Release torque on both arms immediately. No motion.">
          <Button color="red" size="md" leftSection={<IconHandStop size={18} />}
                  onClick={onEStop}>
            EMERGENCY STOP
          </Button>
        </Tooltip>
      </Group>
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        {armCard("left")}
        {armCard("right")}
      </SimpleGrid>
      {s?.last_error && (
        <Alert color="red" mt="sm" icon={<IconAlertTriangleFilled />}>
          {s.last_error}
        </Alert>
      )}
    </Card>
  );
}

function VREndpointCard({ s }: { s: VRStatus | undefined }) {
  const anyConnected = (s?.connected_sides?.length ?? 0) > 0;
  if (!anyConnected) {
    return (
      <Card withBorder padding="md">
        <Text c="dimmed" fz="sm">
          Connect at least one arm to start the VR HTTPS+WSS servers. You'll then see a URL
          to open in your Quest 3 browser.
        </Text>
      </Card>
    );
  }
  if (!s?.vr_endpoint) return null;
  return (
    <Card withBorder padding="md">
      <Stack gap="xs">
        <Text fw={600} fz="sm">Open this on your Quest 3 browser</Text>
        <Group gap="xs">
          <Text className="mono" fz="md">{s.vr_endpoint}</Text>
          <CopyButton value={s.vr_endpoint} timeout={1500}>
            {({ copied, copy }) => (
              <Button size="xs" variant="light"
                      leftSection={copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
                      onClick={copy}>
                {copied ? "copied" : "copy"}
              </Button>
            )}
          </CopyButton>
        </Group>
        <Text fz="xs" c="dimmed">
          Accept the self-signed cert in the Quest browser, enter VR, then come back to this page.
          Squeeze the controller's <b>grip button</b> (middle finger) on the arm you want to drive
          — first press calibrates, then hold to drive.
        </Text>
      </Stack>
    </Card>
  );
}

function EngagementCard({ s, onEngage, busy }: {
  s: VRStatus | undefined;
  onEngage: (engaged: boolean, scale: number, activeArm?: ArmSide) => void;
  busy: boolean;
}) {
  const [localScale, setLocalScale] = useState(0.5);
  useEffect(() => { if (s) setLocalScale(s.scale); }, [s?.scale]);

  const connectedSides = s?.connected_sides ?? [];
  const anyConnected = connectedSides.length > 0;
  const engaged = !!s?.engaged;
  const activeArm = s?.active_arm;

  // When both arms are connected, user must pick which to drive.
  const [localActive, setLocalActive] = useState<ArmSide>("right");
  useEffect(() => {
    if (activeArm) setLocalActive(activeArm);
    else if (connectedSides.length === 1) setLocalActive(connectedSides[0]);
  }, [activeArm, connectedSides.join(",")]);

  const activeCalibrated = activeArm ? s?.arms?.[activeArm]?.calibrated : false;

  return (
    <Card withBorder padding="md"
          style={{ borderColor: engaged ? "var(--mantine-color-red-7)" : undefined,
                   background: engaged ? "rgba(255,77,77,0.06)" : undefined }}>
      <Stack gap="sm">
        <Group justify="space-between" wrap="nowrap">
          <Stack gap={0}>
            <Text fw={700} fz="md">Engagement</Text>
            <Text fz="xs" c="dimmed">
              While ON, VR drives the <b>active</b> arm. The other arm stays torqued
              and holds position. Grip on the active controller anchors + drives.
            </Text>
          </Stack>
          <Switch
            size="xl"
            checked={engaged}
            onChange={(e) => onEngage(e.currentTarget.checked, localScale, localActive)}
            label={engaged ? "ENGAGED" : "Disarmed"}
            color="red"
            disabled={!anyConnected || busy}
            onLabel="ON" offLabel="OFF"
          />
        </Group>

        {connectedSides.length >= 2 && (
          <Group gap="sm">
            <Text fz="sm" fw={600}>Active arm:</Text>
            <SegmentedControl
              size="sm" value={localActive}
              onChange={(v) => {
                const next = v as ArmSide;
                setLocalActive(next);
                if (engaged) onEngage(true, localScale, next);
              }}
              data={[
                { label: "Left", value: "left" },
                { label: "Right", value: "right" },
              ]}
              disabled={busy}
            />
            <Text fz="xs" c="dimmed">only one arm is driven at a time</Text>
          </Group>
        )}

        <Box>
          <Group justify="space-between" mb={2}>
            <Text fz="xs" fw={600}>Speed scale: <span className="mono">{localScale.toFixed(2)}</span></Text>
            <Text fz="xs" c="dimmed">0.5 default (30 cm/s) · 1.0 is 1:1 hand-to-EE · lower for fine work</Text>
          </Group>
          <Slider
            min={0.1} max={1.0} step={0.05} value={localScale}
            onChange={setLocalScale}
            onChangeEnd={(v) => onEngage(engaged, v, localActive)}
            marks={[{ value: 0.1, label: "0.1" }, { value: 0.3, label: "0.3" },
                    { value: 0.5, label: "0.5" }, { value: 1.0, label: "1.0" }]}
            disabled={!anyConnected || busy}
            color={engaged ? "red" : "indigo"}
          />
        </Box>

        <Group gap="xs">
          {activeArm ? (
            <Badge color={activeCalibrated ? "teal" : "gray"} variant="filled">
              {activeCalibrated
                ? `${activeArm} anchored`
                : `${activeArm}: press grip to anchor`}
            </Badge>
          ) : (
            <Badge color="gray" variant="light">no active arm</Badge>
          )}
          {ageBadge(s?.last_goal_age_ms, "last VR goal ")}
          {ageBadge(s?.last_tick_age_ms, "drive tick ")}
        </Group>

        {activeArm && !activeCalibrated && engaged && (
          <Alert color="yellow" icon={<IconAlertTriangleFilled />}>
            VR data flowing for {activeArm} but no anchor captured yet. Squeeze the
            {" "}{activeArm} controller's <b>grip</b> button (middle finger) to anchor
            the controller-to-gripper frame.
          </Alert>
        )}
      </Stack>
    </Card>
  );
}

function JointTable({ side, s }: { side: ArmSide; s: VRStatus | undefined }) {
  const armState = s?.arms?.[side];
  const connected = !!armState?.connected;
  const rows = useMemo(() => {
    if (!connected) return [];
    return JOINTS_ORDER.map((suffix) => {
      const key = `${side}_arm_${suffix}`;
      const present = s?.joint_present[key];
      const target = armState?.joint_target?.[key];
      const bounds = s?.joint_bounds[key] as [number, number] | undefined;
      const nearLow  = bounds && present !== undefined && (present - bounds[0]) < NEAR_BOUND_DEG;
      const nearHigh = bounds && present !== undefined && (bounds[1] - present) < NEAR_BOUND_DEG;
      const warn = nearLow || nearHigh;
      return { suffix, key, present, target, bounds, warn };
    });
  }, [side, connected, s?.joint_present, armState?.joint_target, s?.joint_bounds]);

  return (
    <Card withBorder padding="md">
      <Group justify="space-between" mb="xs">
        <Text fw={600} fz="sm" tt="capitalize">{side} arm</Text>
        {connected ? (
          <Badge size="xs" color={armState?.calibrated ? "teal" : "gray"} variant="light">
            {armState?.calibrated ? "anchored" : "no anchor yet"}
          </Badge>
        ) : (
          <Badge size="xs" color="gray" variant="light">offline</Badge>
        )}
      </Group>
      {!connected && <Text c="dimmed" fz="sm">connect this arm to see joint values</Text>}
      {connected && (
        <Stack gap={2}>
          {rows.map((r) => (
            <Group key={r.suffix} gap="md" wrap="nowrap"
                   style={{
                     padding: "4px 8px", borderRadius: 4,
                     background: r.warn ? "rgba(255,77,77,0.12)" : undefined,
                   }}>
              <Text className="mono" fz="xs" style={{ width: 110 }}>{r.suffix}</Text>
              <Text className="mono" fz="xs" style={{ width: 90, textAlign: "right" }}>
                present <b>{fmt(r.present)}°</b>
              </Text>
              <Text className="mono" fz="xs" style={{ width: 80, textAlign: "right" }}>
                target <b>{fmt(r.target)}°</b>
              </Text>
              <Text className="mono" fz="xs" c="dimmed" style={{ flex: 1 }}>
                [{fmt(r.bounds?.[0])}, {fmt(r.bounds?.[1])}]
              </Text>
            </Group>
          ))}
        </Stack>
      )}
    </Card>
  );
}

/** Calibration card: shows wizard state when a guided calibration is in
 *  progress, otherwise the live anchor/offset/target diagnostics + a button to
 *  start a new calibration. */
function CalibrationCard({ side, armState, onCalibrate, onCancel }: {
  side: ArmSide;
  armState: VRArmState | undefined;
  onCalibrate: (arm: ArmSide) => void;
  onCancel: (arm: ArmSide) => void;
}) {
  const cal = armState?.calibration;
  const connected = !!armState?.connected;
  const anchored = !!armState?.calibrated;
  const wizard = cal?.wizard_state ?? "idle";
  const inWizard = wizard !== "idle";

  // ─── Wizard active: 2-step instructions + live progress ────
  if (cal && inWizard) {
    const motion_cm = (cal.wizard_motion_m * 100).toFixed(1);
    const target_cm = (cal.wizard_target_m * 100).toFixed(0);
    const min_cm = (cal.wizard_min_m * 100).toFixed(0);
    const enough = cal.wizard_motion_m >= cal.wizard_min_m;
    const fwdDone = cal.wizard_fwd_captured;
    const fwdCm = (cal.wizard_last_fwd_m * 100).toFixed(1);

    return (
      <Card withBorder padding="md" style={{ borderColor: "var(--mantine-color-indigo-7)" }}>
        <Group justify="space-between" mb="xs">
          <Text fw={700} fz="md" tt="capitalize">{side} arm — Calibration wizard</Text>
          <Button size="xs" variant="default" onClick={() => onCancel(side)}>
            Cancel
          </Button>
        </Group>

        {/* Progress badges: shows which steps are done */}
        <Group gap={4} mb="sm">
          <Badge size="xs" color={fwdDone ? "teal" : "indigo"} variant={fwdDone ? "filled" : "light"}>
            1. forward {fwdDone ? `✓ ${fwdCm} cm` : ""}
          </Badge>
          <Badge size="xs" color={cal.wizard_up_captured ? "teal" : "indigo"}
                 variant={cal.wizard_up_captured ? "filled" : "light"}>
            2. up {cal.wizard_up_captured ? `✓ ${(cal.wizard_last_up_m * 100).toFixed(1)} cm` : ""}
          </Badge>
          <Badge size="xs" color="indigo"
                 variant={cal.wizard_up_captured ? "filled" : "light"}>
            3. left
          </Badge>
        </Group>

        <Stack gap="sm">
          {/* STEP 1: forward */}
          {wizard === "awaiting_anchor_fwd" && (
            <Alert color="indigo">
              <Text fz="sm" fw={600}>Step 1 of 3 — Capture forward axis</Text>
              <Text fz="xs" mt={4}>
                Put on the headset, stand facing the robot the way you'd
                naturally operate it. Hold the {side} controller at chest
                height.
              </Text>
              <Text fz="xs" mt={4} fw={600}>
                Squeeze the GRIP button on the {side} controller. Then KEEP IT
                HELD and move your hand straight forward.
              </Text>
            </Alert>
          )}
          {wizard === "motioning_fwd" && (
            <>
              <Alert color={enough ? "teal" : "indigo"}>
                <Text fz="sm" fw={600}>Step 1 of 3 — Move FORWARD</Text>
                <Text fz="xs" mt={4}>
                  <b>Keep grip held.</b> Move your hand straight FORWARD (toward
                  the robot, away from your body) ~{target_cm} cm. Then release grip.
                </Text>
              </Alert>
              <Group justify="space-between" mt={4}>
                <Text fz="sm">Motion so far</Text>
                <Badge color={enough ? "teal" : "indigo"} variant="filled">
                  {motion_cm} cm
                </Badge>
              </Group>
              <Slider
                value={cal.wizard_motion_m * 100}
                max={Math.max(cal.wizard_target_m * 100, 1)}
                min={0} disabled
                color={enough ? "teal" : "indigo"}
                marks={[
                  { value: 0, label: "0" },
                  { value: Number(min_cm), label: `min ${min_cm}` },
                  { value: Number(target_cm), label: `${target_cm}` },
                ]}
              />
            </>
          )}

          {/* STEP 2: up */}
          {wizard === "awaiting_anchor_up" && (
            <Alert color="indigo">
              <Text fz="sm" fw={600}>Step 2 of 2 — Capture up axis</Text>
              <Text fz="xs" mt={4}>
                Good — forward axis captured ({fwdCm} cm). Now we'll capture
                "up" so vertical motion maps correctly even if you're sitting
                tilted.
              </Text>
              <Text fz="xs" mt={4} fw={600}>
                Squeeze the GRIP button again. Then KEEP IT HELD and move your
                hand straight UP ~{target_cm} cm.
              </Text>
            </Alert>
          )}
          {wizard === "motioning_up" && (
            <>
              <Alert color={enough ? "teal" : "indigo"}>
                <Text fz="sm" fw={600}>Step 2 of 3 — Move UP</Text>
                <Text fz="xs" mt={4}>
                  <b>Keep grip held.</b> Move your hand straight UP ~{target_cm} cm.
                  Then release grip.
                </Text>
              </Alert>
              <Group justify="space-between" mt={4}>
                <Text fz="sm">Motion so far</Text>
                <Badge color={enough ? "teal" : "indigo"} variant="filled">
                  {motion_cm} cm
                </Badge>
              </Group>
              <Slider
                value={cal.wizard_motion_m * 100}
                max={Math.max(cal.wizard_target_m * 100, 1)}
                min={0} disabled
                color={enough ? "teal" : "indigo"}
                marks={[
                  { value: 0, label: "0" },
                  { value: Number(min_cm), label: `min ${min_cm}` },
                  { value: Number(target_cm), label: `${target_cm}` },
                ]}
              />
            </>
          )}

          {/* STEP 3: left (lateral verification) */}
          {wizard === "awaiting_anchor_left" && (
            <Alert color="indigo">
              <Text fz="sm" fw={600}>Step 3 of 3 — Verify lateral direction</Text>
              <Text fz="xs" mt={4}>
                Forward + up axes captured. Now we verify the lateral sign so
                "your left" actually maps to "robot left" (and not the opposite).
              </Text>
              <Text fz="xs" mt={4} fw={600}>
                Squeeze the GRIP button. KEEP IT HELD and move your hand to
                YOUR LEFT (sideways) ~{target_cm} cm. Then release grip.
              </Text>
            </Alert>
          )}
          {wizard === "motioning_left" && (
            <>
              <Alert color={enough ? "teal" : "indigo"}>
                <Text fz="sm" fw={600}>Step 3 of 3 — Move LEFT</Text>
                <Text fz="xs" mt={4}>
                  <b>Keep grip held.</b> Move your hand to YOUR LEFT (sideways)
                  ~{target_cm} cm. Then release grip — calibration completes
                  automatically and the invert flag is set if the lateral axis
                  is mirrored.
                </Text>
              </Alert>
              <Group justify="space-between" mt={4}>
                <Text fz="sm">Motion so far</Text>
                <Badge color={enough ? "teal" : "indigo"} variant="filled">
                  {motion_cm} cm
                </Badge>
              </Group>
              <Slider
                value={cal.wizard_motion_m * 100}
                max={Math.max(cal.wizard_target_m * 100, 1)}
                min={0} disabled
                color={enough ? "teal" : "indigo"}
                marks={[
                  { value: 0, label: "0" },
                  { value: Number(min_cm), label: `min ${min_cm}` },
                  { value: Number(target_cm), label: `${target_cm}` },
                ]}
              />
            </>
          )}
        </Stack>
      </Card>
    );
  }

  // ─── Wizard idle: show diagnostics + Calibrate button ─────────────────
  const ofs = cal?.offset_robot ?? [0, 0, 0];
  const ofsMag = Math.sqrt(ofs[0] ** 2 + ofs[1] ** 2 + ofs[2] ** 2);
  const lastFwd = cal ? (cal.wizard_last_fwd_m * 100).toFixed(1) : "—";
  const lastUp  = cal ? (cal.wizard_last_up_m  * 100).toFixed(1) : "—";

  return (
    <Card withBorder padding="md">
      <Group justify="space-between" mb="xs">
        <Text fw={600} fz="sm" tt="capitalize">{side} calibration</Text>
        <Group gap={4}>
          <Badge size="xs" color={anchored ? "teal" : "gray"} variant="light">
            {anchored ? "anchored" : "no anchor yet"}
          </Badge>
          <Badge size="xs" color="indigo" variant="light">
            yaw {(cal?.session_yaw_deg ?? 0).toFixed(0)}°
          </Badge>
          {cal?.confidence && (
            <Tooltip label={cal.confidence === "poor"
              ? "Captured motions were too parallel — calibration matrix is shaky. Re-run the wizard with more orthogonal forward/up/left motions."
              : "Captured motions were well-separated; matrix is robust."}>
              <Badge size="xs" variant="light"
                     color={cal.confidence === "good" ? "teal" : "yellow"}>
                confidence: {cal.confidence}
              </Badge>
            </Tooltip>
          )}
          {anchored && (
            <Badge size="xs" variant="light"
                   color={ofsMag < 0.005 ? "gray" : ofsMag < 0.15 ? "teal" : "yellow"}>
              offset {(ofsMag * 100).toFixed(1)} cm
            </Badge>
          )}
        </Group>
      </Group>

      <Group justify="space-between" mb="sm">
        <Text fz="xs" c="dimmed">
          {anchored
            ? "Anchor in place. Offset shows how far your hand has moved from anchor."
            : "No anchor yet. Squeeze grip to start teleop, or click Calibrate to set a new VR→robot frame first."}
        </Text>
        <Button size="xs" variant="filled" color="indigo"
                onClick={() => onCalibrate(side)} disabled={!connected}>
          Calibrate
        </Button>
      </Group>

      {anchored && cal && (
        <Stack gap={2}>
          <Group gap="lg">
            <Text className="mono" fz="xs" c="dimmed" style={{ width: 110 }}>anchor (m)</Text>
            <Text className="mono" fz="xs">
              x={fmt(cal.anchor_ee_pos[0], 3)}, y={fmt(cal.anchor_ee_pos[1], 3)},
              z={fmt(cal.anchor_ee_pos[2], 3)}
            </Text>
          </Group>
          <Group gap="lg">
            <Text className="mono" fz="xs" c="dimmed" style={{ width: 110 }}>offset (m)</Text>
            <Text className="mono" fz="xs">
              x={fmt(ofs[0], 3)}, y={fmt(ofs[1], 3)}, z={fmt(ofs[2], 3)}
            </Text>
          </Group>
          <Group gap="lg">
            <Text className="mono" fz="xs" c="dimmed" style={{ width: 110 }}>target (m)</Text>
            <Text className="mono" fz="xs">
              x={fmt(cal.target_ee_pos[0], 3)}, y={fmt(cal.target_ee_pos[1], 3)},
              z={fmt(cal.target_ee_pos[2], 3)}
            </Text>
          </Group>
        </Stack>
      )}
      {cal && (cal.wizard_last_fwd_m > 0 || cal.wizard_last_up_m > 0) && (
        <Text fz="xs" c="dimmed" mt="xs">
          Last calibration: forward={lastFwd} cm, up={lastUp} cm.
        </Text>
      )}
      {cal?.persisted?.saved && (
        <Group gap={4} mt={4}>
          <Badge size="xs" color="teal" variant="light">saved</Badge>
          <Text fz="xs" c="dimmed">
            Calibration persisted to <span className="mono">config/vr_calibration.yaml</span>
            {cal.persisted.calibrated_at && ` at ${cal.persisted.calibrated_at.replace("T", " ")}`}
            . Will auto-load next session.
          </Text>
        </Group>
      )}
      {cal?.confidence === "poor" && (
        <Alert color="yellow" mt="xs" icon={<IconAlertTriangleFilled />}>
          <Text fz="xs">
            Calibration confidence is <b>poor</b>: your forward/up/left motions
            were too parallel for the matrix to be reliable. Click <b>Calibrate</b>{" "}
            again and make each motion more distinctly perpendicular — forward
            should clearly differ from up, and left from forward.
          </Text>
        </Alert>
      )}
    </Card>
  );
}

function ControllerCard({ side, armState }: { side: ArmSide; armState: VRArmState | undefined }) {
  const c = armState?.controller;
  const pos = c?.position;
  const rot = c?.rotation;
  return (
    <Card withBorder padding="md">
      <Group justify="space-between" mb="xs">
        <Text fw={600} fz="sm" tt="capitalize">{side} controller</Text>
        <Group gap={4}>
          {ageBadge(c?.age_ms, "age ")}
          <Badge variant="light" color={c?.trigger ? "red" : "gray"}>
            trigger {c?.trigger ? "ON" : "off"}
          </Badge>
          <Badge variant="light" color="indigo">
            mode {c?.mode || "—"}
          </Badge>
        </Group>
      </Group>
      <Stack gap={4}>
        <Group gap="lg">
          <Text className="mono" fz="xs" c="dimmed">rel.position</Text>
          <Text className="mono" fz="xs">
            x={fmt(pos?.[0], 3)} m, y={fmt(pos?.[1], 3)} m, z={fmt(pos?.[2], 3)} m
          </Text>
        </Group>
        <Group gap="lg">
          <Text className="mono" fz="xs" c="dimmed">rotation (quat)</Text>
          <Text className="mono" fz="xs">
            x={fmt(rot?.[0], 3)}, y={fmt(rot?.[1], 3)},
            z={fmt(rot?.[2], 3)}, w={fmt(rot?.[3], 3)}
          </Text>
        </Group>
      </Stack>
    </Card>
  );
}

/** Home-pose card: per arm, shows whether a home is captured + joint values +
 *  Capture / Go-to-Home buttons. Capture reads the current joint positions and
 *  writes them to config/xlerobot.yaml. Go-to-Home slow-interpolates the arm
 *  back to that saved pose using the drive loop's per-tick caps + KP. */
function HomeCard({ side, armState, onCapture, onGoHome, onCancelHome,
                     onTorqueRelease, onTorqueLock, busy }: {
  side: ArmSide;
  armState: VRArmState | undefined;
  onCapture: (arm: ArmSide) => void;
  onGoHome: (arm: ArmSide) => void;
  onCancelHome: (arm: ArmSide) => void;
  onTorqueRelease: (arm: ArmSide) => void;
  onTorqueLock: (arm: ArmSide) => void;
  busy: boolean;
}) {
  const connected = !!armState?.connected;
  const torqueOn = armState?.torque_enabled ?? true;
  const home = armState?.home;
  const captured = !!home?.captured;
  const homing = !!home?.homing;
  const joints = home?.joints || {};
  return (
    <Card withBorder padding="md"
          style={{
            borderColor: !torqueOn ? "var(--mantine-color-orange-7)" : undefined,
            background: !torqueOn ? "rgba(255, 165, 0, 0.06)" : undefined,
          }}>
      <Group justify="space-between" mb="xs">
        <Group gap="xs">
          <IconHome size={16} />
          <Text fw={600} fz="sm" tt="capitalize">{side} home pose</Text>
          <Badge size="xs" color={captured ? "teal" : "gray"} variant="light">
            {captured ? "captured" : "not captured"}
          </Badge>
          {!torqueOn && (
            <Badge size="xs" color="orange" variant="filled">torque OFF</Badge>
          )}
          {homing && <Badge size="xs" color="orange" variant="filled">homing…</Badge>}
        </Group>
        <Group gap={4}>
          {torqueOn ? (
            <Tooltip label="Disable torque so you can hand-pose the arm. The arm will go limp — support it!">
              <Button size="xs" leftSection={<IconLockOpen size={14} />}
                      color="orange" variant="light"
                      onClick={() => onTorqueRelease(side)}
                      disabled={busy || !connected || homing}>
                Release for posing
              </Button>
            </Tooltip>
          ) : (
            <Tooltip label="Re-enable torque at the CURRENT position (no snap-back)">
              <Button size="xs" leftSection={<IconLock size={14} />}
                      color="teal" variant="filled"
                      onClick={() => onTorqueLock(side)}
                      disabled={busy || !connected}>
                Lock at current
              </Button>
            </Tooltip>
          )}
          <Tooltip label="Read present joints and write to config/xlerobot.yaml. Works whether torque is on or off — but if torque was off, lock it after.">
            <Button size="xs" leftSection={<IconHomeCog size={14} />}
                    color="indigo" variant="light"
                    onClick={() => onCapture(side)}
                    disabled={busy || !connected}>
              Capture
            </Button>
          </Tooltip>
          {!homing ? (
            <Tooltip label="Slowly interpolate back to the saved home pose">
              <Button size="xs" leftSection={<IconHome size={14} />}
                      color="teal" variant="filled"
                      onClick={() => onGoHome(side)}
                      disabled={busy || !connected || !captured || !torqueOn}>
                Go to Home
              </Button>
            </Tooltip>
          ) : (
            <Button size="xs" color="gray" variant="default"
                    onClick={() => onCancelHome(side)}>
              Cancel homing
            </Button>
          )}
        </Group>
      </Group>

      {!torqueOn && (
        <Alert color="orange" mb="xs" icon={<IconAlertTriangleFilled />}>
          <Text fz="xs">
            Torque on {side} arm is RELEASED. The arm is limp — support it by
            hand so it doesn't sag under gravity. Pose it to where you want,
            then click <b>Capture</b> (writes to YAML), then <b>Lock at
            current</b> (re-enables torque holding at that pose).
          </Text>
        </Alert>
      )}

      {!captured && (
        <Text fz="xs" c="dimmed">
          Click <b>Release for posing</b> → hand-pose the arm → click
          <b> Capture</b> → click <b>Lock at current</b>. This writes the joint
          angles to <span className="mono">config/xlerobot.yaml</span> and lets
          you reliably return to this pose with <b>Go to Home</b>.
        </Text>
      )}
      {captured && (
        <Stack gap={2}>
          {JOINTS_ORDER.map((suffix) => {
            const k = `${side}_arm_${suffix}`;
            return (
              <Group key={suffix} gap="md" wrap="nowrap">
                <Text className="mono" fz="xs" style={{ width: 110 }}>{suffix}</Text>
                <Text className="mono" fz="xs" style={{ width: 90, textAlign: "right" }}>
                  saved <b>{fmt(joints[k], 2)}°</b>
                </Text>
              </Group>
            );
          })}
        </Stack>
      )}
    </Card>
  );
}

/** Recording toggle + per-episode task description. Mirrors the B-button on
 *  right Quest controller. Writes LeRobotDataset v2 with action +
 *  observation.state + cameras. The `task` string is stored on every frame
 *  and used as conditioning for VLA training. */
function RecordingCard({ s, task, onTaskChange, storageRoot, onStorageRootChange,
                          onToggle, busy }: {
  s: VRStatus | undefined;
  task: string;
  onTaskChange: (t: string) => void;
  storageRoot: string;
  onStorageRootChange: (path: string) => void;
  onToggle: (enabled: boolean) => void;
  busy: boolean;
}) {
  const recording = !!s?.recording;
  const info = s?.recording_info;
  const taskOk = task.trim().length > 0;
  // Compute the effective placeholder: whatever the backend says will be used
  // if the user leaves the field blank.
  const defaultRoot = info?.root || "~/.cache/huggingface/lerobot/<repo_id>/";
  return (
    <Card withBorder padding="md"
          style={{ borderColor: recording ? "var(--mantine-color-grape-7)" : undefined,
                   background: recording ? "rgba(190, 70, 200, 0.06)" : undefined }}>
      <Stack gap="sm">
        <Group justify="space-between" wrap="nowrap">
          <Group gap="xs">
            <Text fw={700} fz="md">Dataset recording</Text>
            {recording ? (
              <Badge color="grape" variant="filled">REC</Badge>
            ) : (
              <Badge color="gray" variant="light">idle</Badge>
            )}
            {info?.episodes_saved !== undefined && (
              <Badge color="gray" variant="light">
                {info.episodes_saved} episode{info.episodes_saved === 1 ? "" : "s"} saved
              </Badge>
            )}
            {recording && info?.frames_in_current_episode !== undefined && (
              <Badge color="grape" variant="light">
                {info.frames_in_current_episode} frames
              </Badge>
            )}
          </Group>
          <Button
            color={recording ? "gray" : "grape"}
            variant={recording ? "default" : "filled"}
            leftSection={recording ? <IconPlayerStop size={16} /> : <IconPlayerPlay size={16} />}
            onClick={() => onToggle(!recording)}
            disabled={busy || (!recording && !taskOk)}
          >
            {recording ? "Stop recording" : "Start recording"}
          </Button>
        </Group>

        <TextInput
          label="Task description"
          description="Natural-language instruction this episode demonstrates. Required for LeRobot v2 — stored on every frame and used as conditioning for VLA training. Example: 'Pick the red block and place it in the bin'."
          placeholder="Pick the red block and place it in the bin"
          value={task}
          onChange={(e) => onTaskChange(e.currentTarget.value)}
          disabled={recording}
          error={!taskOk && !recording ? "task description required before starting an episode" : false}
        />

        <TextInput
          label="Storage path"
          description="Absolute filesystem path where episodes are written. Leave blank to use the HuggingFace default. The path is captured on the FIRST Start recording — change requires backend restart or EMERGENCY STOP."
          placeholder={defaultRoot}
          value={storageRoot}
          onChange={(e) => onStorageRootChange(e.currentTarget.value)}
          disabled={recording || (info?.episodes_saved ?? 0) > 0}
        />

        <Text fz="xs" c="dimmed">
          Press <b>B</b> on the right Quest controller, or click <b>Start recording</b>, to
          begin an episode. Both arms' commanded + present joints and every camera with a
          role assigned on the Cameras page are captured. Writes LeRobot v2 to{" "}
          <span className="mono">{info?.root || defaultRoot}</span> · repo_id{" "}
          <span className="mono">{info?.repo_id || "(configure dataset.repo_id)"}</span>.
        </Text>
      </Stack>
    </Card>
  );
}

function CameraStrip() {
  const tiles = ["head", "left_wrist", "right_wrist"] as const;
  return (
    <Card withBorder padding="md">
      <Text fw={600} fz="sm" mb="xs">Cameras</Text>
      <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="sm">
        {tiles.map((t) => (
          <div key={t} className="cam-tile" style={{ aspectRatio: "4 / 3" }}>
            <img alt={t} src={`/camera/${t}/stream`} />
          </div>
        ))}
      </SimpleGrid>
    </Card>
  );
}

export function VRTeleop() {
  const { data: s, mutate, isLoading } = useSWR<VRStatus>(
    "/api/vr/status", fetcher, { refreshInterval: 200 },
  );
  const [busy, setBusy] = useState(false);
  const [eStopConfirm, setEStopConfirm] = useState(false);
  // Per-episode task description. Sent to the recorder on every Start; LeRobot
  // v2 stores it on each frame and uses it as conditioning input for VLA training.
  const [taskDescription, setTaskDescription] = useState("");
  // Dataset storage root override (empty = use backend's effective default).
  // Locked-in on first Start recording; subsequent edits ignored until the
  // backend restarts or EMERGENCY STOP destroys the recorder.
  const [storageRoot, setStorageRoot] = useState("");
  // Pre-fill from the backend's cached last_task on first non-empty status load,
  // so a page refresh doesn't wipe what the user already typed.
  const [taskInitialized, setTaskInitialized] = useState(false);
  useEffect(() => {
    if (taskInitialized) return;
    const cached = s?.recording_info?.last_task;
    if (cached) {
      setTaskDescription(cached);
      setTaskInitialized(true);
    } else if (s) {
      // Status loaded but no cached task — mark initialized so we don't
      // overwrite the user's typing later.
      setTaskInitialized(true);
    }
  }, [s, taskInitialized]);

  const handleConnect = async (arm: ArmSide) => {
    setBusy(true);
    try {
      await api.vrConnect(arm);
      notifications.show({ color: "teal", title: `Connected (${arm} arm)`,
        message: "Torque enabled. Open the VR endpoint URL on the Quest." });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Connect failed", message: String(e) });
    } finally { setBusy(false); }
  };

  const handleDisconnect = async (arm: ArmSide) => {
    setBusy(true);
    try {
      await api.vrDisconnect(arm);
      notifications.show({ color: "yellow", title: `Disconnected (${arm})`,
        message: "Torque released. No motion was taken." });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Disconnect failed", message: String(e) });
    } finally { setBusy(false); }
  };

  const handleEStop = async () => {
    setEStopConfirm(false);
    setBusy(true);
    try {
      await api.vrEmergencyStop();
      notifications.show({ color: "red", title: "EMERGENCY STOP",
        message: "Torque released. Robot will not move." });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "E-stop failed", message: String(e) });
    } finally { setBusy(false); }
  };

  const handleEngage = async (engaged: boolean, scale: number, activeArm?: ArmSide) => {
    try {
      await api.vrEngage(engaged, scale, activeArm);
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Engage failed", message: String(e) });
    }
  };

  const handleRecordingToggle = async (enabled: boolean) => {
    try {
      // Only send task on START — on STOP, task is irrelevant.
      await api.vrSetRecording(
        enabled,
        enabled ? taskDescription.trim() : "",
        enabled ? storageRoot.trim() : "",
      );
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Recording toggle failed",
                            message: String(e) });
    }
  };

  const handleCalibrateStart = async (arm: ArmSide) => {
    try {
      await api.vrCalibrateStart(arm);
      notifications.show({
        color: "indigo",
        title: `Calibrating ${arm} arm`,
        message: "Put on the headset and follow the on-screen wizard.",
      });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Calibration failed to start",
                            message: String(e) });
    }
  };

  const handleCalibrateCancel = async (arm: ArmSide) => {
    try {
      await api.vrCalibrateCancel(arm);
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Cancel failed", message: String(e) });
    }
  };

  const handleHomeCapture = async (arm: ArmSide) => {
    setBusy(true);
    try {
      await api.vrHomeCapture(arm);
      notifications.show({
        color: "teal",
        title: `${arm} home captured`,
        message: "Joint angles written to config/xlerobot.yaml.",
      });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Capture failed", message: String(e) });
    } finally { setBusy(false); }
  };

  const handleGoHome = async (arm: ArmSide) => {
    try {
      await api.vrHomeGo(arm);
      notifications.show({
        color: "indigo",
        title: `${arm} homing…`,
        message: "Arm is interpolating to the saved home pose.",
      });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Go-home failed", message: String(e) });
    }
  };

  const handleCancelHome = async (arm: ArmSide) => {
    try {
      await api.vrHomeCancel(arm);
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Cancel homing failed", message: String(e) });
    }
  };

  const handleTorqueRelease = async (arm: ArmSide) => {
    setBusy(true);
    try {
      await api.vrTorqueRelease(arm);
      notifications.show({
        color: "orange",
        title: `${arm} torque RELEASED`,
        message: "Arm is limp — support it by hand. Click 'Lock at current' when done.",
      });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Release failed", message: String(e) });
    } finally { setBusy(false); }
  };

  const handleTorqueLock = async (arm: ArmSide) => {
    setBusy(true);
    try {
      await api.vrTorqueLock(arm);
      notifications.show({
        color: "teal",
        title: `${arm} torque LOCKED`,
        message: "Arm is now holding at its current position.",
      });
      await mutate();
    } catch (e) {
      notifications.show({ color: "red", title: "Lock failed", message: String(e) });
    } finally { setBusy(false); }
  };

  return (
    <Container size="xl" px={0}>
      <Group justify="space-between" mb="md">
        <Stack gap={0}>
          <Title order={2} fw={600}>
            <Group gap="xs"><IconHeadset size={26} /> VR Teleop</Group>
          </Title>
          <Text c="dimmed" fz="sm">
            Bimanual Meta Quest 3 teleop, engage-gated: both arms can be torqued
            simultaneously but only one is driven by VR at a time. Buttons:
            <b> grip</b> (middle finger) = calibrate + drive;{" "}
            <b>trigger</b> (index) = gripper;{" "}
            <b>A</b> (right) / <b>X</b> (left) = toggle engage for that arm;{" "}
            <b>B</b> (right) = toggle dataset recording. Safety: engage toggle,
            grip-press calibration, stale-goal watchdog. No motion on disconnect
            or shutdown.
          </Text>
        </Stack>
      </Group>

      <Stack gap="md">
        <ConnectionBar
          s={s}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          onEStop={() => setEStopConfirm(true)}
          busy={busy || isLoading}
        />
        <VREndpointCard s={s} />
        <EngagementCard s={s} onEngage={handleEngage} busy={busy} />
        <RecordingCard s={s}
                       task={taskDescription}
                       onTaskChange={setTaskDescription}
                       storageRoot={storageRoot}
                       onStorageRootChange={setStorageRoot}
                       onToggle={handleRecordingToggle}
                       busy={busy} />

        <Grid>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <HomeCard side="left" armState={s?.arms?.left}
                       onCapture={handleHomeCapture}
                       onGoHome={handleGoHome}
                       onCancelHome={handleCancelHome}
                       onTorqueRelease={handleTorqueRelease}
                       onTorqueLock={handleTorqueLock}
                       busy={busy} />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <HomeCard side="right" armState={s?.arms?.right}
                       onCapture={handleHomeCapture}
                       onGoHome={handleGoHome}
                       onCancelHome={handleCancelHome}
                       onTorqueRelease={handleTorqueRelease}
                       onTorqueLock={handleTorqueLock}
                       busy={busy} />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}><JointTable side="left" s={s} /></Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}><JointTable side="right" s={s} /></Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <CalibrationCard side="left" armState={s?.arms?.left}
                              onCalibrate={handleCalibrateStart}
                              onCancel={handleCalibrateCancel} />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <CalibrationCard side="right" armState={s?.arms?.right}
                              onCalibrate={handleCalibrateStart}
                              onCancel={handleCalibrateCancel} />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ControllerCard side="left" armState={s?.arms?.left} />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <ControllerCard side="right" armState={s?.arms?.right} />
          </Grid.Col>
        </Grid>

        <CameraStrip />
      </Stack>

      <Modal opened={eStopConfirm} onClose={() => setEStopConfirm(false)}
             title="Emergency stop?" centered size="sm">
        <Stack>
          <Alert color="red" icon={<IconAlertTriangleFilled />}>
            Torque will be released on BOTH arms immediately. The robot will NOT
            move to a safe pose — it will hold whatever pose it's in (or sag
            under gravity if it can't hold itself). Use only if something is wrong.
          </Alert>
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setEStopConfirm(false)}>Cancel</Button>
            <Button color="red" onClick={handleEStop}>Stop now</Button>
          </Group>
        </Stack>
      </Modal>
    </Container>
  );
}
