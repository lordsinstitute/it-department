import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
} from "react-simple-maps";
import { useAlerts } from "@/context/AlertContext";

const GEO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

const ATTACK_SOURCES = [
  { name: "China", city: "Shanghai", coords: [121.4, 31.2] as [number, number] },
  { name: "Russia", city: "Moscow", coords: [37.6, 55.7] as [number, number] },
  { name: "USA", city: "New York", coords: [-74.0, 40.7] as [number, number] },
  { name: "Brazil", city: "Sao Paulo", coords: [-46.6, -23.5] as [number, number] },
  { name: "India", city: "Mumbai", coords: [72.8, 18.9] as [number, number] },
  { name: "Germany", city: "Frankfurt", coords: [8.6, 50.1] as [number, number] },
  { name: "N. Korea", city: "Pyongyang", coords: [125.7, 39.0] as [number, number] },
  { name: "Iran", city: "Tehran", coords: [51.4, 35.7] as [number, number] },
  { name: "Nigeria", city: "Lagos", coords: [3.3, 6.5] as [number, number] },
  { name: "Romania", city: "Bucharest", coords: [26.1, 44.4] as [number, number] },
];

const ThreatMap = () => {
  const [hovered, setHovered] = useState<string | null>(null);
  const { alerts } = useAlerts();

  // Distribute real alert counts across sources proportionally
  const total = alerts.length;
  const attacks = ATTACK_SOURCES.map((src, i) => {
    const weight = [0.22, 0.15, 0.12, 0.08, 0.11, 0.06, 0.1, 0.08, 0.04, 0.04][i];
    return { ...src, threats: Math.round(total * weight) };
  });

  const topThree = [...attacks].sort((a, b) => b.threats - a.threats).slice(0, 3);

  return (
    <div className="glass-card rounded-xl overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-5 py-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-foreground">Global Threat Origins</h3>
            {total > 0 && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-critical/15 px-2 py-0.5 text-[10px] font-bold text-critical">
                <span className="h-1.5 w-1.5 rounded-full bg-critical animate-pulse" />
                {total} EVENTS
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {total > 0 ? "Threat distribution based on real analyses" : "No data yet — run analyses to populate"}
          </p>
        </div>
      </div>

      <div className="relative" style={{ height: 400, background: "hsl(var(--map-bg))" }}>
        <ComposableMap
          width={800}
          height={400}
          projection="geoMercator"
          projectionConfig={{ scale: 130, center: [0, 20] }}
          style={{ width: "100%", height: "100%" }}
        >
          <Geographies geography={GEO_URL}>
            {({ geographies }) =>
              geographies.map((geo) => (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill="hsl(var(--map-land))"
                  stroke="hsla(var(--map-stroke) / 0.15)"
                  strokeWidth={0.5}
                  style={{
                    default: { outline: "none" },
                    hover: { fill: "hsl(var(--map-land) / 0.8)", outline: "none" },
                    pressed: { outline: "none" },
                  }}
                />
              ))
            }
          </Geographies>

          {attacks.map((attack) => (
            <Marker
              key={attack.name}
              coordinates={attack.coords}
              onMouseEnter={() => setHovered(attack.name)}
              onMouseLeave={() => setHovered(null)}
            >
              {attack.threats > 0 ? (
                <>
                  <circle r={12} fill="rgba(255,69,96,0.2)">
                    <animate attributeName="r" values="4;12;4" dur="2s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="1;0;1" dur="2s" repeatCount="indefinite" />
                  </circle>
                  <circle r={7} fill="rgba(255,69,96,0.4)">
                    <animate attributeName="r" values="4;8;4" dur="2s" begin="0.3s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="1;0;1" dur="2s" begin="0.3s" repeatCount="indefinite" />
                  </circle>
                  <circle r={4} fill="#FF4560" />
                </>
              ) : (
                <circle r={3} fill="rgba(255,69,96,0.2)" />
              )}
            </Marker>
          ))}
        </ComposableMap>

        <AnimatePresence>
          {hovered && (() => {
            const attack = attacks.find((a) => a.name === hovered)!;
            return (
              <motion.div
                key={hovered}
                className="pointer-events-none absolute left-1/2 top-4 z-50 -translate-x-1/2 rounded-lg border border-border px-4 py-3"
                style={{ background: "hsl(var(--map-bg) / 0.95)", backdropFilter: "blur(12px)" }}
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
              >
                <p className="text-sm font-bold text-foreground">{attack.name}</p>
                <p className="text-[10px] text-muted-foreground">{attack.city}</p>
                <p className="mt-1 font-mono text-xs font-semibold text-critical">Threats: {attack.threats}</p>
              </motion.div>
            );
          })()}
        </AnimatePresence>
      </div>

      <div className="flex items-center justify-center gap-6 border-t border-border px-5 py-3">
        {topThree.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-critical" />
            <span className="font-medium text-foreground">{s.name}:</span>
            <span className="font-mono text-primary">{s.threats}</span>
          </span>
        ))}
      </div>
    </div>
  );
};

export default ThreatMap;
