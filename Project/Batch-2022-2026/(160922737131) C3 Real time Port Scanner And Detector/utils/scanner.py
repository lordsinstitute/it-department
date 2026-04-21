import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# In-memory jobs (demo-friendly). For enterprise, you'd persist jobs.
_JOBS = {}
_LOCK = threading.Lock()

COMMON_PORTS = [
    20, 21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 465, 587,
    993, 995, 1433, 1521, 2049, 3306, 3389, 5432, 5900, 6379, 8080, 8443
]

def _parse_ports(port_mode: str, ports_text: str):
    port_mode = (port_mode or "common").lower()
    ports_text = (ports_text or "").strip()

    if port_mode == "common":
        return sorted(set(COMMON_PORTS))

    if port_mode == "top100":
        # Minimal “top” selection without external datasets
        base = set(COMMON_PORTS)
        extra = {1, 7, 9, 37, 69, 79, 88, 111, 113, 119, 161, 389, 636, 873, 1723, 27017}
        return sorted(base.union(extra))

    if port_mode == "range":
        # Expect "start-end"
        try:
            start, end = ports_text.split("-", 1)
            start = int(start.strip())
            end = int(end.strip())
            start = max(1, start)
            end = min(65535, end)
            if start > end:
                start, end = end, start
            return list(range(start, end + 1))
        except Exception:
            return sorted(set(COMMON_PORTS))

    if port_mode == "custom":
        # Expect "22,80,443"
        ports = set()
        for part in ports_text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                p = int(part)
                if 1 <= p <= 65535:
                    ports.add(p)
            except Exception:
                continue
        return sorted(ports) if ports else sorted(set(COMMON_PORTS))

    return sorted(set(COMMON_PORTS))

def start_scan_job(targets, port_mode, ports_text, timeout_ms, max_threads, requested_by="unknown"):
    job_id = str(uuid.uuid4())
    ports = _parse_ports(port_mode, ports_text)

    job = {
        "job_id": job_id,
        "state": "running",
        "message": "Scan started",
        "requested_by": requested_by,
        "started_at": time.time(),
        "progress": {
            "targets_total": len(targets),
            "targets_done": 0,
            "ports_total": len(ports),
            "ports_scanned": 0,
            "open_found": 0,
        },
        "result": None,
        "error": None
    }

    with _LOCK:
        _JOBS[job_id] = job

    t = threading.Thread(
        target=_run_scan,
        args=(job_id, targets, ports, timeout_ms, max_threads),
        daemon=True
    )
    t.start()
    return job_id

def get_job_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return {"state": "error", "message": "Job not found."}
        # return a safe copy
        return {
            "job_id": job["job_id"],
            "state": job["state"],
            "message": job["message"],
            "progress": job["progress"],
            "result": job["result"],
            "error": job["error"],
        }

def _tcp_connect_banner(host: str, port: int, timeout: float):
    """
    TCP connect scan + light banner grab (safe, minimal).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        res = s.connect_ex((host, port))
        if res != 0:
            return False, None

        banner = None
        try:
            # Try small read; some services send greeting (FTP/SMTP/SSH)
            s.settimeout(0.6)
            banner = s.recv(128)
            if banner:
                banner = banner.decode(errors="ignore").strip()
            else:
                banner = None
        except Exception:
            banner = None

        return True, banner
    finally:
        try:
            s.close()
        except Exception:
            pass

def _run_scan(job_id: str, targets, ports, timeout_ms: int, max_threads: int):
    timeout = max(0.2, min(5.0, timeout_ms / 1000.0))
    max_threads = max(20, min(800, max_threads))

    all_results = []
    alerts = []

    def update_progress(**kwargs):
        with _LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return
            job["progress"].update(kwargs)

    try:
        for ti, target in enumerate(targets, start=1):
            # Resolve DNS early (reduce noise)
            resolved_ip = None
            try:
                resolved_ip = socket.gethostbyname(target)
            except Exception:
                resolved_ip = None

            target_result = {
                "target": target,
                "resolved_ip": resolved_ip,
                "open_ports": [],
                "closed_count": 0,
                "errors": [],
            }

            # Thread pool per target
            open_count = 0
            scanned = 0

            with ThreadPoolExecutor(max_workers=max_threads) as ex:
                futures = {}
                for p in ports:
                    futures[ex.submit(_tcp_connect_banner, target, p, timeout)] = p

                for fut in as_completed(futures):
                    p = futures[fut]
                    scanned += 1
                    update_progress(ports_scanned=(_JOBS[job_id]["progress"]["ports_scanned"] + 1))

                    try:
                        is_open, banner = fut.result()
                        if is_open:
                            open_count += 1
                            target_result["open_ports"].append({
                                "port": p,
                                "banner": banner
                            })
                    except Exception as e:
                        target_result["errors"].append(f"Port {p}: {str(e)}")

            target_result["closed_count"] = len(ports) - len(target_result["open_ports"])

            # Alert simulation (terminal)
            for item in target_result["open_ports"]:
                if item["port"] in (23, 445, 3389, 5900, 1433, 3306):
                    alerts.append({
                        "type": "ALERT",
                        "target": target,
                        "port": item["port"],
                        "message": "Potentially risky service port detected (simulation)."
                    })

            all_results.append(target_result)

            with _LOCK:
                job = _JOBS.get(job_id)
                if job:
                    job["progress"]["targets_done"] = ti
                    job["progress"]["open_found"] += len(target_result["open_ports"])
                    job["message"] = f"Scanning {ti}/{len(targets)} targets..."

        result = {
            "scan_type": "TCP Connect (Real-time demo)",
            "targets_display": ", ".join(targets[:5]) + ("..." if len(targets) > 5 else ""),
            "targets_total": len(targets),
            "ports_scanned_per_target": len(ports),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results,
            "alerts": alerts,
            "notes": [
                "Alerts are simulated (terminal + UI toast) per project constraints.",
                "Banner grabbing is best-effort and may be empty for many services."
            ]
        }

        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["state"] = "done"
                job["message"] = "Scan complete"
                job["result"] = result

        # Terminal alert simulation
        if alerts:
            print("\n=== ALERT SIMULATION ===")
            for a in alerts[:20]:
                print(f"[{a['type']}] {a['target']}:{a['port']} -> {a['message']}")
            print("========================\n")

    except Exception as e:
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["state"] = "error"
                job["message"] = "Scan failed"
                job["error"] = str(e)