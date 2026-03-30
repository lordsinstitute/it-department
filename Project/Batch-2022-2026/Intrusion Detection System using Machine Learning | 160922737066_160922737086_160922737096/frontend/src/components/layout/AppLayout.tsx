import { useEffect, useRef, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import AppSidebar from "./AppSidebar";
import AppHeader from "./AppHeader";
import ThreatBanner from "./ThreatBanner";
import ThreatTicker from "../dashboard/ThreatTicker";
import MiniChatbot from "../analyze/MiniChatbot";
import { useIsMobile } from "@/hooks/use-mobile";

const pageVariants = {
  initial: { opacity: 0, y: 20, filter: "blur(4px)" },
  animate: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.45, ease: [0.25, 0.46, 0.45, 0.94] } },
  exit: { opacity: 0, y: -10, filter: "blur(4px)", transition: { duration: 0.25 } },
};

const AppLayout = () => {
  const location = useLocation();
  const isMobile = useIsMobile();
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const mainRef = useRef<HTMLElement>(null);

  // Scroll to top on route change
  useEffect(() => {
    mainRef.current?.scrollTo(0, 0);
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen bg-background cyber-grid">
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <AppSidebar
        collapsed={!sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        isMobile={isMobile}
      />
      <div className={`flex min-w-0 flex-1 flex-col transition-all duration-300 ${
        isMobile ? "ml-0" : sidebarOpen ? "ml-60" : "ml-16"
      }`}>
        <AppHeader onMenuToggle={() => setSidebarOpen(!sidebarOpen)} isMobile={isMobile} />
        <ThreatBanner />
        <ThreatTicker />
        <main ref={mainRef} className="flex-1 p-3 sm:p-6 overflow-x-hidden overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      {location.pathname !== "/ai-assistant" && <MiniChatbot />}
    </div>
  );
};

export default AppLayout;
