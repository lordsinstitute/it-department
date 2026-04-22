import re
from collections import Counter

from utils.scoring import clamp


SUSPICIOUS_PORTS = {
    21: "FTP (cleartext auth)",
    23: "Telnet (cleartext)",
    25: "SMTP (phishing/exfil potential)",
    53: "DNS (tunneling potential)",
    110: "POP3 (cleartext variants)",
    143: "IMAP (cleartext variants)",
    445: "SMB (lateral movement target)",
    3389: "RDP (brute force target)",
    4444: "Common reverse shell port",
    8080: "Common proxy/alt HTTP",
    9001: "Alt services / backdoors",
}

ATTACK_KEYWORDS = [
    ("nmap", "Recon tool indicator"),
    ("masscan", "High-speed port scan indicator"),
    ("metasploit", "Exploit framework indicator"),
    ("mimikatz", "Credential dump indicator"),
    ("powershell -enc", "Encoded PowerShell indicator"),
    ("cmd.exe /c", "Command execution indicator"),
    ("invoke-webrequest", "Suspicious download indicator"),
    ("curl http", "Suspicious download indicator"),
    ("wget http", "Suspicious download indicator"),
    ("base64", "Encoding/tunneling hint"),
    ("user-agent:", "Header analysis hint"),
    ("authorization:", "Credential leakage hint"),
]

IP_REGEX = r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
DOMAIN_REGEX = r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"
PORT_REGEX = r"\b(?:port\s*[:=]?\s*(\d{1,5})|\:(\d{1,5})\b)"


def analyze_payload(text: str) -> dict:
    """
    Analyzes pasted text or uploaded text logs that resemble:
    - tcpdump output
    - firewall logs
    - proxy logs
    - netstat output
    - application network logs

    This does NOT do raw packet capture (driverless).
    """
    raw = text or ""
    lowered = raw.lower()

    findings = []
    score = 0

    # 1) Extract IPs / domains
    ips = re.findall(IP_REGEX, raw)
    domains = re.findall(DOMAIN_REGEX, raw)
    ip_counts = Counter(ips)
    domain_counts = Counter([d.lower() for d in domains])

    top_ips = ip_counts.most_common(5)
    top_domains = domain_counts.most_common(5)

    # 2) Suspicious keywords
    kw_hits = []
    for kw, desc in ATTACK_KEYWORDS:
        if kw in lowered:
            kw_hits.append({"keyword": kw, "reason": desc})
    if kw_hits:
        findings.append({
            "title": "Suspicious keywords detected",
            "severity": "Medium",
            "details": kw_hits[:20],
            "recommendation": "Review the payload/log source for malicious tooling, encoded commands, or suspicious downloads."
        })
        score += min(25, 5 * len(kw_hits))

    # 3) Detect potential scanning by many ports or many destination attempts
    ports = []
    for m in re.finditer(PORT_REGEX, raw, flags=re.IGNORECASE):
        p1, p2 = m.group(1), m.group(2)
        p = p1 or p2
        if p:
            try:
                pv = int(p)
                if 1 <= pv <= 65535:
                    ports.append(pv)
            except ValueError:
                pass

    port_counts = Counter(ports)
    distinct_ports = len(port_counts)

    if distinct_ports >= 20:
        findings.append({
            "title": "Possible port scanning behavior",
            "severity": "High",
            "details": {
                "distinct_ports": distinct_ports,
                "top_ports": port_counts.most_common(10),
            },
            "recommendation": "Check for scanning tools, block suspicious sources, and validate with endpoint telemetry."
        })
        score += 35
    elif distinct_ports >= 10:
        findings.append({
            "title": "Elevated port diversity observed",
            "severity": "Medium",
            "details": {
                "distinct_ports": distinct_ports,
                "top_ports": port_counts.most_common(10),
            },
            "recommendation": "Review whether this traffic pattern is expected for the host/app."
        })
        score += 18

    # 4) Suspicious ports seen
    sus_port_hits = []
    for p, count in port_counts.items():
        if p in SUSPICIOUS_PORTS:
            sus_port_hits.append({"port": p, "count": count, "note": SUSPICIOUS_PORTS[p]})
    if sus_port_hits:
        sev = "High" if any(x["port"] in (445, 3389, 4444) for x in sus_port_hits) else "Medium"
        findings.append({
            "title": "Suspicious / high-risk ports referenced",
            "severity": sev,
            "details": sorted(sus_port_hits, key=lambda x: (-x["count"], x["port"]))[:15],
            "recommendation": "Confirm service legitimacy. Restrict exposure and monitor authentication attempts."
        })
        score += 10 + min(20, len(sus_port_hits) * 3)

    # 5) DNS tunneling heuristics (very long labels / base64-ish)
    long_tokens = re.findall(r"\b[a-zA-Z0-9+/=_-]{50,}\b", raw)
    if len(long_tokens) >= 3 and ("dns" in lowered or ":53" in lowered or " 53 " in lowered):
        findings.append({
            "title": "Possible DNS tunneling indicators",
            "severity": "High",
            "details": {
                "sample_tokens": long_tokens[:5],
                "token_count": len(long_tokens),
            },
            "recommendation": "Inspect DNS logs, block suspicious domains, and check for data exfil patterns."
        })
        score += 30

    # 6) Credential leakage hints (very simple)
    cred_patterns = []
    if "authorization:" in lowered:
        cred_patterns.append("Authorization header present")
    if re.search(r"\bpassword\s*[:=]", lowered):
        cred_patterns.append("Password key/value pattern present")
    if re.search(r"\bapi[_-]?key\s*[:=]", lowered):
        cred_patterns.append("API key key/value pattern present")
    if cred_patterns:
        findings.append({
            "title": "Potential credential exposure",
            "severity": "High",
            "details": cred_patterns,
            "recommendation": "Remove secrets from logs, rotate credentials, and use a secrets manager."
        })
        score += 28

    # 7) Summaries
    summary = {
        "characters_analyzed": len(raw),
        "lines_analyzed": raw.count("\n") + 1 if raw else 0,
        "unique_ips": len(ip_counts),
        "unique_domains": len(domain_counts),
        "distinct_ports": distinct_ports,
        "keyword_hits": len(kw_hits),
    }

    # 8) Indicators block (shown on UI)
    indicators = {
        "top_ips": top_ips,
        "top_domains": top_domains,
        "top_ports": port_counts.most_common(10),
    }

    # Clamp score 0..100
    score = clamp(score, 0, 100)

    return {
        "score": score,
        "summary": summary,
        "findings": findings,
        "indicators": indicators,
    }