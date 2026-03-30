import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/context/AuthContext";
import { AlertProvider } from "@/context/AlertContext";
import ProtectedRoute from "@/components/ProtectedRoute";
import AdminRoute from "@/components/AdminRoute";
import AppLayout from "@/components/layout/AppLayout";
import Overview from "./pages/Overview";
import Analyze from "./pages/Analyze";
import AlertLog from "./pages/AlertLog";
import Performance from "./pages/Performance";
import Notifications from "./pages/Notifications";
import Settings from "./pages/Settings";
import AIAssistant from "./pages/AIAssistant";
import AIHistory from "./pages/AIHistory";
import AdminDashboard from "./pages/AdminDashboard";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <AuthProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route
              element={
                <ProtectedRoute>
                  <AlertProvider>
                    <AppLayout />
                  </AlertProvider>
                </ProtectedRoute>
              }
            >
              <Route path="/" element={<Overview />} />
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/alerts" element={<AlertLog />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/notifications" element={<Notifications />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/ai-assistant" element={<AIAssistant />} />
              <Route path="/ai-history" element={<AIHistory />} />
              <Route path="/admin" element={<AdminRoute><AdminDashboard /></AdminRoute>} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </AuthProvider>
  </QueryClientProvider>
);

export default App;
