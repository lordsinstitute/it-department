def alert_simulation(risk_level: str, analysis: dict) -> None:
    """
    NO email / NO external config.
    Shows an "alert simulation" in terminal. Also usable in UI via flash (handled in routes).
    """
    victim = (analysis.get("victim_candidate") or {}).get("dst_ip", "unknown")
    pkt = (analysis.get("victim_candidate") or {}).get("packets", 0)
    sources = (analysis.get("victim_candidate") or {}).get("unique_sources", 0)
    score = analysis.get("risk_score", 0)

    print("\n" + "=" * 70)
    print("ALERT SIMULATION (NO external integrations)")
    print(f"Risk Level : {risk_level}")
    print(f"Victim     : {victim}")
    print(f"Packets    : {pkt}")
    print(f"Sources    : {sources}")
    print(f"Risk Score : {score}")
    print("Action     : Investigate traffic, rate-limit, enable WAF/ACLs, capture PCAP.")
    print("=" * 70 + "\n")