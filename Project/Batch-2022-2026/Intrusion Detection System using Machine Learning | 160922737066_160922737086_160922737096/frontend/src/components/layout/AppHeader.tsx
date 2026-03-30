import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Bell, Radio, LogOut, User, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/AuthContext";
import { useAlerts } from "@/context/AlertContext";

const routeNames: Record<string, string> = {
  "/": "Dashboard",
  "/analyze": "Analyze Threats",
  "/alerts": "Alert History",
  "/performance": "ML Performance",
  "/notifications": "Email Alerts",
  "/settings": "Settings",
  "/ai-assistant": "AI Assistant",
  "/ai-history": "AI History",
  "/admin": "Admin Panel",
};

interface AppHeaderProps {
  onMenuToggle?: () => void;
  isMobile?: boolean;
}

const AppHeader = ({ onMenuToggle, isMobile }: AppHeaderProps) => {
  const location = useLocation();
  const [time, setTime] = useState(new Date());
  const { user, signOut, isAdmin } = useAuth();
  const { criticalCount } = useAlerts();
  const navigate = useNavigate();

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const pageName = routeNames[location.pathname] || "Dashboard";

  return (
    <header className="sticky top-0 z-40 flex items-center justify-between border-b border-border bg-background/80 px-3 sm:px-6 py-2.5 sm:py-3 backdrop-blur-md gap-2">
      {/* Left: Menu + Breadcrumb */}
      <div className="flex items-center gap-2 text-sm min-w-0">
        {isMobile && (
          <button onClick={onMenuToggle} className="rounded-md p-1.5 hover:bg-secondary/50 transition-colors mr-1">
            <Menu className="h-5 w-5 text-muted-foreground" />
          </button>
        )}
        <span className="text-muted-foreground hidden sm:inline">IDS Shield</span>
        <span className="text-muted-foreground hidden sm:inline">/</span>
        <span className="font-medium text-foreground truncate text-xs sm:text-sm">{pageName}</span>
      </div>

      {/* Center: Pulsing Status - hidden on mobile */}
      <div className="hidden md:flex items-center gap-3">
        <div className="flex items-center gap-2">
          <Radio className="h-3 w-3 text-success animate-pulse-glow" />
          <span className="text-xs font-medium text-success">System Active</span>
        </div>
        <span className="text-[9px] text-muted-foreground">·</span>
        <span className="text-[9px] font-medium text-primary">Powered by Random Forest ML — 99.27% accuracy</span>
      </div>

      {/* Right: User + Clock + Bell + Logout */}
      <div className="flex items-center gap-1.5 sm:gap-3 flex-shrink-0">
        {user && !isMobile && (
          <div className="hidden lg:flex items-center gap-2 rounded-lg bg-secondary/50 px-3 py-1.5">
            <User className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-xs font-medium text-foreground truncate max-w-[150px]">
              {user.email}
            </span>
            {isAdmin && (
              <span className="text-[9px] font-bold text-critical bg-critical/10 rounded px-1.5 py-0.5">ADMIN</span>
            )}
          </div>
        )}
        <span className="font-mono text-[10px] sm:text-xs text-muted-foreground hidden sm:inline">
          {time.toLocaleTimeString("en-US", { hour12: false })}
        </span>
        <button
          className="relative cursor-pointer rounded-md p-1.5 hover:bg-secondary/50 transition-colors"
          onClick={() => navigate("/notifications")}
          aria-label="Notifications"
        >
          <Bell className="h-4 w-4 text-muted-foreground" />
          {criticalCount > 0 && (
            <div className="absolute -right-0.5 -top-0.5 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-critical animate-pulse">
              <span className="text-[8px] font-bold text-critical-foreground">{criticalCount}</span>
            </div>
          )}
        </button>
        <Button
          variant="ghost"
          size="sm"
          onClick={signOut}
          className="h-7 sm:h-8 gap-1 sm:gap-1.5 text-xs text-muted-foreground hover:text-critical px-1.5 sm:px-2"
        >
          <LogOut className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Logout</span>
        </Button>
      </div>
    </header>
  );
};

export default AppHeader;
