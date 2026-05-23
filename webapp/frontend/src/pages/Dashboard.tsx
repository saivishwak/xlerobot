import {
  Alert, Badge, Button, Card, Code, Container, Grid, Group, SimpleGrid, Stack,
  Text, Title, Loader, Tooltip,
} from "@mantine/core";
import {
  IconCircleCheckFilled, IconAlertTriangleFilled, IconCircleXFilled, IconInfoCircleFilled,
  IconRefresh,
} from "@tabler/icons-react";
import useSWR from "swr";
import { CheckStatus, DoctorCheck, fetcher } from "../api";

const COLOR: Record<CheckStatus, string> = {
  ok: "teal", warn: "yellow", fail: "red", info: "blue",
};

function StatusIcon({ status }: { status: CheckStatus }) {
  const Icon = {
    ok: IconCircleCheckFilled, warn: IconAlertTriangleFilled,
    fail: IconCircleXFilled,   info: IconInfoCircleFilled,
  }[status];
  return <Icon size={18} color={`var(--mantine-color-${COLOR[status]}-5)`} />;
}

function Tally({ checks }: { checks: DoctorCheck[] }) {
  const counts: Record<CheckStatus, number> = { ok: 0, warn: 0, fail: 0, info: 0 };
  for (const c of checks) counts[c.status]++;
  return (
    <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md">
      {(["ok", "warn", "fail", "info"] as CheckStatus[]).map((s) => (
        <Card withBorder key={s} padding="md">
          <Group justify="space-between">
            <Stack gap={0}>
              <Text fz="xs" c="dimmed" tt="uppercase" fw={600}>{s}</Text>
              <Text fz={28} fw={700}>{counts[s]}</Text>
            </Stack>
            <StatusIcon status={s} />
          </Group>
        </Card>
      ))}
    </SimpleGrid>
  );
}

function CheckRow({ check }: { check: DoctorCheck }) {
  return (
    <Card withBorder padding="sm" radius="md">
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="md">
        <Group gap="sm" align="flex-start" wrap="nowrap" style={{ minWidth: 0, flex: 1 }}>
          <StatusIcon status={check.status} />
          <Stack gap={2} style={{ minWidth: 0, flex: 1 }}>
            <Text fw={600} fz="sm">{check.name}</Text>
            <Code className="mono" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {check.detail}
            </Code>
          </Stack>
        </Group>
        <Badge variant="light" color={COLOR[check.status]}>{check.status}</Badge>
      </Group>
    </Card>
  );
}

export function Dashboard() {
  const { data, error, isLoading, mutate } = useSWR<{ checks: DoctorCheck[] }>(
    "/api/doctor", fetcher, { refreshInterval: 0 },
  );
  return (
    <Container size="xl" px={0}>
      <Group justify="space-between" mb="lg">
        <Stack gap={0}>
          <Title order={2} fw={600}>Diagnostics</Title>
          <Text c="dimmed" fz="sm">
            System health: USB devices, V4L2 video nodes, serial ports, configured motor/camera
            paths, and required Python modules.
          </Text>
        </Stack>
        <Tooltip label="Re-run all checks">
          <Button leftSection={<IconRefresh size={16} />} onClick={() => mutate()} loading={isLoading}>
            Run checks
          </Button>
        </Tooltip>
      </Group>

      {error && (
        <Alert color="red" mb="md" title="Failed to load checks">
          {String(error)}
        </Alert>
      )}

      {data && (
        <Stack gap="md">
          <Tally checks={data.checks} />
          <Grid>
            {data.checks.map((c, i) => (
              <Grid.Col span={{ base: 12, md: 6 }} key={i}>
                <CheckRow check={c} />
              </Grid.Col>
            ))}
          </Grid>
        </Stack>
      )}

      {isLoading && !data && (
        <Group justify="center" mt="xl">
          <Loader />
        </Group>
      )}
    </Container>
  );
}
