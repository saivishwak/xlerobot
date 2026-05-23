import {
  Alert, Badge, Card, Container, Group, Select, SimpleGrid, Stack,
  Text, Title, Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import useSWR from "swr";
import { CameraSpec, api, fetcher } from "../api";

function CameraTile({ cam, roles, onAssign }: {
  cam: CameraSpec;
  roles: string[];
  onAssign: (cam: CameraSpec, role: string | null) => Promise<void>;
}) {
  const [errored, setErrored] = useState(false);
  const [busy, setBusy] = useState(false);
  const handleChange = async (val: string | null) => {
    setBusy(true);
    try { await onAssign(cam, val); } finally { setBusy(false); }
  };
  return (
    <Card withBorder padding="md" radius="md">
      <Group justify="space-between" mb="xs">
        <Stack gap={0} style={{ minWidth: 0 }}>
          <Group gap={6}>
            <Text fw={600} fz="sm">{cam.name}</Text>
            {cam.role && <Badge size="xs" variant="light" color="indigo">{cam.role}</Badge>}
          </Group>
          <Text fz="xs" c="dimmed" style={{ wordBreak: "break-all" }}>
            {cam.card || "—"}
          </Text>
          <Text fz="xs" c="dimmed" className="mono" style={{ wordBreak: "break-all" }}>
            {cam.by_path || cam.path}
          </Text>
        </Stack>
      </Group>

      <div className="cam-tile">
        {!errored ? (
          <img
            src={`/camera/${encodeURIComponent(cam.name)}/stream?t=${cam.path}`}
            alt={cam.name}
            onError={() => setErrored(true)}
          />
        ) : (
          <div className="placeholder">stream unavailable — device may be in use</div>
        )}
      </div>

      <Tooltip label={cam.by_path ? "Assign this device to a robot role" : "no /dev/v4l/by-path symlink"}
               disabled={!!cam.by_path}>
        <Select
          mt="md"
          label="Role"
          size="xs"
          data={[{ value: "", label: "(none)" }, ...roles.map(r => ({ value: r, label: r }))]}
          value={cam.role || ""}
          onChange={(v) => handleChange(v || null)}
          disabled={!cam.by_path || busy}
        />
      </Tooltip>
    </Card>
  );
}

export function Cameras() {
  const { data, error, mutate } = useSWR<{ cameras: CameraSpec[]; roles: string[] }>(
    "/api/cameras", fetcher,
  );

  const handleAssign = async (cam: CameraSpec, role: string | null) => {
    if (!cam.by_path) return;
    try {
      await api.assign(cam.by_path, role);
      await mutate();
      notifications.show({
        color: "teal",
        title: role ? `Assigned → ${role}` : "Cleared assignment",
        message: cam.by_path,
      });
    } catch (e) {
      notifications.show({ color: "red", title: "Assign failed", message: String(e) });
    }
  };

  return (
    <Container size="xl" px={0}>
      <Stack gap={0} mb="lg">
        <Title order={2} fw={600}>Cameras</Title>
        <Text c="dimmed" fz="sm">
          Live MJPEG previews of every capture-capable V4L2 device. Assign each USB camera to
          a robot role (<Text span fw={600}>head</Text>, <Text span fw={600}>left_wrist</Text>,{" "}
          <Text span fw={600}>right_wrist</Text>) using the dropdown — changes save to{" "}
          <Text span className="mono">config/xlerobot.yaml</Text> immediately.
        </Text>
      </Stack>

      {error && <Alert color="red" mb="md" title="Failed to load cameras">{String(error)}</Alert>}

      {data && (
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
          {data.cameras.map((c) => (
            <CameraTile key={c.name} cam={c} roles={data.roles} onAssign={handleAssign} />
          ))}
          {data.cameras.length === 0 && (
            <Text c="dimmed" ta="center" py="xl">
              No capture-capable cameras detected. Plug in USB cameras and refresh.
            </Text>
          )}
        </SimpleGrid>
      )}
    </Container>
  );
}
