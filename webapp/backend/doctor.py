"""System checks: USB devices, video nodes, serial ports, package imports."""
from __future__ import annotations

import importlib
import os
import pathlib
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"

Status = Literal["ok", "warn", "fail", "info"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} not installed"
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"


def _load_config() -> dict:
    if not CONFIG_YAML.is_file():
        return {}
    try:
        return yaml.safe_load(CONFIG_YAML.read_text()) or {}
    except Exception as e:
        return {"_parse_error": str(e)}


def check_lsusb() -> Check:
    rc, out, err = _run(["lsusb"])
    if rc == 127:
        return Check("lsusb", "warn", "lsusb not installed (apt install usbutils)")
    if rc != 0:
        return Check("lsusb", "fail", err.strip() or "lsusb failed")
    devices = [ln for ln in out.splitlines() if ln.strip()]
    return Check("lsusb", "ok", f"{len(devices)} devices\n" + "\n".join(devices))


def check_video_nodes() -> Check:
    nodes = sorted(pathlib.Path("/dev").glob("video*"))
    by_path_dir = pathlib.Path("/dev/v4l/by-path")
    by_path = sorted(by_path_dir.iterdir()) if by_path_dir.is_dir() else []
    detail_lines = [f"raw nodes: {', '.join(p.name for p in nodes) or '(none)'}"]
    if by_path:
        detail_lines.append("by-path:")
        detail_lines.extend(f"  {p.name} -> {os.readlink(p)}" for p in by_path)
    else:
        detail_lines.append("by-path: (empty)")
    status: Status = "ok" if nodes else "warn"
    return Check("video devices", status, "\n".join(detail_lines))


def check_serial_ports() -> Check:
    serial_dir = pathlib.Path("/dev/serial/by-id")
    by_id = sorted(serial_dir.iterdir()) if serial_dir.is_dir() else []
    tty_usb = sorted(pathlib.Path("/dev").glob("ttyUSB*"))
    tty_acm = sorted(pathlib.Path("/dev").glob("ttyACM*"))
    detail = [
        f"/dev/ttyUSB*: {', '.join(p.name for p in tty_usb) or '(none)'}",
        f"/dev/ttyACM*: {', '.join(p.name for p in tty_acm) or '(none)'}",
    ]
    if by_id:
        detail.append("by-id:")
        detail.extend(f"  {p.name} -> {os.readlink(p)}" for p in by_id)
    status: Status = "ok" if (tty_usb or tty_acm or by_id) else "warn"
    return Check("serial ports", status, "\n".join(detail))


def check_serial_access() -> Check:
    """Verify the current user can actually read+write the configured motor ports."""
    cfg = _load_config()
    r = (cfg.get("robot") or {})
    ports = [p for p in (r.get("port_left_base"), r.get("port_right_head")) if p]
    if not ports:
        return Check("serial port access", "info",
                     "no motor ports set in config/xlerobot.yaml")
    lines = []
    bad: list[str] = []
    for p in ports:
        path = pathlib.Path(p)
        if not path.exists():
            lines.append(f"{p}: missing")
            bad.append(p)
            continue
        ok_r = os.access(p, os.R_OK)
        ok_w = os.access(p, os.W_OK)
        st = path.stat()
        try:
            import grp, pwd
            owner = pwd.getpwuid(st.st_uid).pw_name
            group = grp.getgrgid(st.st_gid).gr_name
        except Exception:
            owner = str(st.st_uid); group = str(st.st_gid)
        perms = oct(st.st_mode & 0o777)
        if not (ok_r and ok_w):
            bad.append(p)
        lines.append(f"{p}: {owner}:{group} {perms}  read={ok_r} write={ok_w}")

    if bad:
        try:
            import grp, pwd
            uname = pwd.getpwuid(os.geteuid()).pw_name
            # Groups the user has *configured* in /etc/group (what `id` would show after re-login).
            configured = {g.gr_name for g in grp.getgrall() if uname in g.gr_mem}
            configured.add(grp.getgrgid(pwd.getpwnam(uname).pw_gid).gr_name)
            # Groups *this process* actually has right now.
            current = {grp.getgrgid(gid).gr_name for gid in os.getgroups()}

            lines.append("")
            lines.append(f"current user: {uname}")
            lines.append(f"groups (this process): {', '.join(sorted(current))}")

            if "dialout" not in configured:
                lines.append("")
                lines.append(f"Fix: sudo usermod -aG dialout {uname}")
                lines.append("Then log out + back in (or run `newgrp dialout`).")
            elif "dialout" not in current:
                lines.append("")
                lines.append(f"{uname} IS in dialout per /etc/group, but THIS process was started")
                lines.append("before the membership took effect — supplementary groups are frozen at login.")
                lines.append("")
                lines.append("Fix one of:")
                lines.append("  1) log out and log back in (then restart this server)")
                lines.append("  2) in a fresh shell: newgrp dialout && make webapp-backend")
        except Exception:
            pass
        return Check("serial port access", "fail", "\n".join(lines))
    return Check("serial port access", "ok", "\n".join(lines))


def check_config_paths() -> list[Check]:
    cfg = _load_config()
    if "_parse_error" in cfg:
        return [Check("config/xlerobot.yaml", "fail", cfg["_parse_error"])]
    if not cfg:
        return [Check("config/xlerobot.yaml", "fail", f"not found: {CONFIG_YAML}")]

    checks: list[Check] = []

    robot = cfg.get("robot", {})
    for key in ("port_left_base", "port_right_head"):
        port = robot.get(key)
        if not port:
            checks.append(Check(f"motor port: {key}", "warn", "not set in config"))
            continue
        if pathlib.Path(port).exists():
            checks.append(Check(f"motor port: {key}", "ok", port))
        else:
            checks.append(Check(f"motor port: {key}", "fail", f"missing: {port}"))

    for name, c in (cfg.get("cameras") or {}).items():
        if c.get("type") == "opencv":
            path = c.get("path", "")
            if path and pathlib.Path(path).exists():
                checks.append(Check(f"camera: {name}", "ok", path))
            else:
                checks.append(Check(f"camera: {name}", "fail", f"missing: {path or '(no path)'}"))
        elif c.get("type") == "realsense":
            checks.append(Check(f"camera: {name}", "info",
                                f"realsense serial={c.get('serial', '?')} (check with rs-enumerate-devices)"))
        else:
            checks.append(Check(f"camera: {name}", "warn", f"unknown type: {c.get('type')}"))
    return checks


def check_python_imports() -> list[Check]:
    targets = [
        ("lerobot", "lerobot"),
        ("lerobot.robots.xlerobot", "lerobot.robots.xlerobot"),
        ("XLeVR.vr_monitor", "XLeVR.vr_monitor"),
        ("openpi_client", "openpi_client"),
        ("cv2", "cv2"),
        ("pyrealsense2", "pyrealsense2"),
    ]
    out: list[Check] = []
    for label, mod in targets:
        try:
            importlib.import_module(mod)
            out.append(Check(f"import: {label}", "ok", "available"))
        except Exception as e:
            status: Status = "info" if label == "pyrealsense2" else "fail"
            out.append(Check(f"import: {label}", status, str(e)))
    return out


def check_v4l2_caps() -> Check:
    if shutil.which("v4l2-ctl") is None:
        return Check("v4l2-ctl", "info", "v4l2-ctl not installed (apt install v4l-utils)")
    rc, out, err = _run(["v4l2-ctl", "--list-devices"])
    return Check("v4l2-ctl --list-devices", "ok" if rc == 0 else "warn",
                 (out or err).strip() or "(no devices)")


def check_realsense() -> Check:
    if shutil.which("rs-enumerate-devices"):
        rc, out, _ = _run(["rs-enumerate-devices", "-s"])
        return Check("realsense", "ok" if rc == 0 else "info",
                     out.strip() or "no realsense devices")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devs = [{"name": d.get_info(rs.camera_info.name),
                 "serial": d.get_info(rs.camera_info.serial_number)} for d in ctx.devices]
        return Check("realsense (pyrealsense2)", "ok" if devs else "info",
                     "\n".join(f"{d['name']} ({d['serial']})" for d in devs) or "no devices")
    except Exception as e:
        return Check("realsense", "info", f"not available: {e}")


def run_doctor() -> list[dict]:
    checks: list[Check] = []
    checks.append(check_lsusb())
    checks.append(check_video_nodes())
    checks.append(check_serial_ports())
    checks.append(check_serial_access())
    checks.extend(check_config_paths())
    checks.extend(check_python_imports())
    checks.append(check_v4l2_caps())
    checks.append(check_realsense())
    return [asdict(c) for c in checks]
