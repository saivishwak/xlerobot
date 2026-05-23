import {
  AppShell, Burger, Group, NavLink, ScrollArea, Text, Title, Badge,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import {
  IconStethoscope, IconCamera, IconHeadset, IconRobot,
} from "@tabler/icons-react";
import { NavLink as RRNavLink, Route, Routes, useLocation } from "react-router-dom";
import useSWR from "swr";
import { VRStatus, api, fetcher } from "./api";
import { Dashboard } from "./pages/Dashboard";
import { Cameras }   from "./pages/Cameras";
import { VRTeleop }  from "./pages/VRTeleop";

const NAV = [
  { to: "/",          label: "VR Teleop", icon: IconHeadset },
  { to: "/diagnostics", label: "Diagnostics", icon: IconStethoscope },
  { to: "/cameras",   label: "Cameras",   icon: IconCamera },
] as const;

function GlobalStatus() {
  const vr = useSWR<VRStatus>("/api/vr/status", fetcher, { refreshInterval: 1500 });
  const s = vr.data;
  if (!s) return null;

  const armColor = s.arm ? "indigo" : "gray";
  const engagedColor =
    s.engaged ? "red" :
    s.connected ? "yellow" :
    "gray";
  return (
    <Group gap="xs">
      <Badge variant={s.connected ? "filled" : "light"} color={armColor}>
        {s.connected ? `${s.arm} arm · connected` : "no arm"}
      </Badge>
      <Badge variant={s.engaged ? "filled" : "light"} color={engagedColor}>
        {s.engaged ? "ENGAGED · VR live" : s.connected ? "armed · safe" : "idle"}
      </Badge>
    </Group>
  );
}

export function App() {
  const [opened, { toggle }] = useDisclosure();
  const loc = useLocation();
  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 220, breakpoint: "sm", collapsed: { mobile: !opened } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <IconRobot size={22} />
            <Title order={4} fw={600}>XLeRobot Console</Title>
          </Group>
          <GlobalStatus />
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="sm">
        <ScrollArea>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              component={RRNavLink}
              to={n.to}
              label={n.label}
              leftSection={<n.icon size={18} stroke={1.6} />}
              active={loc.pathname === n.to}
            />
          ))}
          <Text c="dimmed" fz="xs" mt="md" px="xs">v0.2.0 · safe-vr</Text>
        </ScrollArea>
      </AppShell.Navbar>

      <AppShell.Main>
        <Routes>
          <Route path="/"             element={<VRTeleop />} />
          <Route path="/diagnostics"  element={<Dashboard />} />
          <Route path="/cameras"      element={<Cameras />} />
        </Routes>
      </AppShell.Main>
    </AppShell>
  );
}

void api;
