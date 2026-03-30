import { Shield, LayoutDashboard, Search, FileText, BarChart3, Bell, Settings, User, PanelLeftClose, PanelLeft, X, Bot, History, ShieldAlert } from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";

const navItems = [
  { title: "Dashboard", url: "/", icon: LayoutDashboard },
  { title: "AI Assistant", url: "/ai-assistant", icon: Bot },
  { title: "Analyze Threats", url: "/analyze", icon: Search },
  { title: "Alert History", url: "/alerts", icon: FileText },
  { title: "AI History", url: "/ai-history", icon: History },
  { title: "ML Performance", url: "/performance", icon: BarChart3 },
  { title: "Email Alerts", url: "/notifications", icon: Bell },
  { title: "Settings", url: "/settings", icon: Settings },
];

const adminItems = [
  { title: "Admin Panel", url: "/admin", icon: ShieldAlert },
];

interface AppSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  isMobile?: boolean;
}

const AppSidebar = ({ collapsed, onToggle, isMobile }: AppSidebarProps) => {
  const location = useLocation();
  const { user, isAdmin, role } = useAuth();

  const allItems = isAdmin ? [...navItems, ...adminItems] : navItems;

  const renderNavItems = (items: typeof navItems, onClick?: () => void) =>
    items.map((item) => {
      const isActive = location.pathname === item.url;
      const isAdminItem = adminItems.some((a) => a.url === item.url);
      return (
        <NavLink
          key={item.title}
          to={item.url}
          end
          className={`group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
            isActive
              ? isAdminItem ? "bg-critical/10 text-critical" : "bg-primary/10 text-primary"
              : "text-muted-foreground hover:bg-secondary hover:text-foreground"
          } ${collapsed && !isMobile ? "justify-center" : ""}`}
          activeClassName=""
          title={collapsed && !isMobile ? item.title : undefined}
          onClick={onClick}
        >
          {isActive && (
            <div className={`absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2 rounded-r-full ${
              isAdminItem ? "bg-critical" : "bg-primary glow-primary"
            }`} />
          )}
          <item.icon className="h-4 w-4 flex-shrink-0" />
          {(isMobile || !collapsed) && <span>{item.title}</span>}
        </NavLink>
      );
    });

  // Mobile drawer
  if (isMobile) {
    if (collapsed) return null;
    return (
      <aside className="fixed left-0 top-0 z-50 flex h-screen w-64 flex-col border-r border-border bg-[hsl(var(--sidebar-background))] shadow-2xl">
        <div className="flex items-center justify-between border-b border-border px-3 py-5">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-primary/10 p-2 glow-primary flex-shrink-0">
              <Shield className="h-6 w-6 text-primary" />
            </div>
            <div className="min-w-0">
              <h1 className="text-base font-bold tracking-tight text-foreground truncate">IDS Shield</h1>
              <p className="text-[10px] font-medium text-primary truncate">Adaptive IDS</p>
            </div>
          </div>
          <button onClick={onToggle} className="rounded-lg p-2 text-muted-foreground hover:bg-secondary">
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 space-y-1 px-2 py-4 overflow-y-auto">
          {renderNavItems(navItems, onToggle)}
          {isAdmin && (
            <>
              <div className="my-2 mx-3 border-t border-border" />
              <p className="px-3 text-[9px] uppercase tracking-wider text-critical font-bold mb-1">Admin</p>
              {renderNavItems(adminItems, onToggle)}
            </>
          )}
        </nav>

        <div className="border-t border-border px-3 py-4">
          <div className="flex items-center gap-3">
            <div className="relative flex-shrink-0">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-secondary">
                <User className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-[hsl(var(--sidebar-background))] bg-success" />
            </div>
            <div>
              <p className="text-xs font-medium text-foreground truncate max-w-[150px]">{user?.email?.split("@")[0] || "User"}</p>
              <div className="flex items-center gap-1.5">
                <p className="text-[10px] text-success">Online</p>
                {isAdmin && <span className="text-[9px] font-bold text-critical bg-critical/10 rounded px-1">ADMIN</span>}
              </div>
            </div>
          </div>
        </div>
      </aside>
    );
  }

  // Desktop sidebar
  return (
    <aside className={`fixed left-0 top-0 z-50 flex h-screen flex-col border-r border-border bg-[hsl(var(--sidebar-background))] transition-all duration-300 ${collapsed ? "w-16" : "w-60"}`}>
      <div className="flex items-center gap-3 border-b border-border px-3 py-5 overflow-hidden">
        <div className="rounded-lg bg-primary/10 p-2 glow-primary flex-shrink-0">
          <Shield className="h-6 w-6 text-primary" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <h1 className="text-base font-bold tracking-tight text-foreground truncate">IDS Shield</h1>
            <p className="text-[10px] font-medium text-primary truncate">Adaptive Intrusion Detection System</p>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-2 py-4 overflow-y-auto">
        {renderNavItems(navItems)}
        {isAdmin && (
          <>
            <div className="my-2 mx-3 border-t border-border" />
            {!collapsed && <p className="px-3 text-[9px] uppercase tracking-wider text-critical font-bold mb-1">Admin</p>}
            {renderNavItems(adminItems)}
          </>
        )}
      </nav>

      <div className="border-t border-border px-2 py-2">
        <button
          onClick={onToggle}
          className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          {!collapsed && <span className="text-xs">Collapse</span>}
        </button>
      </div>

      <div className="border-t border-border px-3 py-4">
        <div className={`flex items-center gap-3 ${collapsed ? "justify-center" : ""}`}>
          <div className="relative flex-shrink-0">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-secondary">
              <User className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-[hsl(var(--sidebar-background))] bg-success" />
          </div>
          {!collapsed && (
            <div>
              <p className="text-xs font-medium text-foreground truncate max-w-[120px]">{user?.email?.split("@")[0] || "User"}</p>
              <div className="flex items-center gap-1.5">
                <p className="text-[10px] text-success">Online</p>
                {isAdmin && <span className="text-[9px] font-bold text-critical bg-critical/10 rounded px-1">ADMIN</span>}
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

export default AppSidebar;
