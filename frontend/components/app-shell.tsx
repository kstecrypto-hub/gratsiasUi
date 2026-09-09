"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { api, messageFromError } from "@/lib/api";
import type { Administrator } from "@/lib/types";
import { ErrorState, LoadingState } from "@/components/page-state";
import { setActiveTimezone } from "@/lib/format";
import { EvaluationAvailability } from "@/components/evaluation-availability";

const navigation = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/analyze", label: "Analyze Calls" },
  { href: "/results", label: "Results" },
  { href: "/keywords", label: "Keywords" },
  { href: "/operators", label: "Operators" },
  { href: "/settings", label: "Settings" },
] as const;

function pageTitle(pathname: string): string {
  if (pathname === "/evaluation" || pathname.startsWith("/evaluation/")) return "Evaluation";
  if (pathname.startsWith("/processing/")) return "Processing";
  if (pathname.startsWith("/calls/")) return "Call detail";
  return navigation.find((item) => pathname === item.href)?.label || "Yeastar Call Analyzer";
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<Administrator | null>();
  const [error, setError] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("");
  const [evaluationEnabled, setEvaluationEnabled] = useState(false);

  const loadUser = useCallback(async () => {
    setError("");
    try {
      const currentUser = await api.auth.me();
      try {
        setEvaluationEnabled((await api.features()).evaluation_ui_enabled === true);
      } catch {
        setEvaluationEnabled(false);
      }
      try {
        const applicationSettings = await api.settings.get();
        setActiveTimezone(applicationSettings.default_timezone);
      } catch {
        // Authentication remains usable if optional display settings cannot be loaded.
      }
      setUser(currentUser);
    } catch (caught) {
      if (caught instanceof Error && "status" in caught && (caught as { status?: number }).status === 401) {
        router.replace("/login");
        return;
      }
      setError(messageFromError(caught));
      setUser(null);
    }
  }, [router]);

  useEffect(() => {
    void loadUser();
    const expired = () => router.replace("/login?expired=1");
    window.addEventListener("session-expired", expired);
    return () => window.removeEventListener("session-expired", expired);
  }, [loadUser, router]);

  useEffect(() => setMenuOpen(false), [pathname]);
  useEffect(() => {
    if (!pathname.startsWith("/processing/")) setProcessingStatus("");
    const update = (event: Event) => setProcessingStatus((event as CustomEvent<string>).detail || "");
    window.addEventListener("processing-status", update);
    return () => window.removeEventListener("processing-status", update);
  }, [pathname]);

  async function logout() {
    setLoggingOut(true);
    try {
      await api.auth.logout();
    } finally {
      router.replace("/login");
      router.refresh();
    }
  }

  if (user === undefined) return <LoadingState label="Checking your session" />;
  if (!user && error) return <ErrorState message={error} onRetry={() => void loadUser()} />;
  if (!user) return <LoadingState label="Opening sign in" />;

  return (
    <div className="app-shell">
      <aside id="main-navigation" className={`sidebar${menuOpen ? " open" : ""}`} aria-label="Main navigation">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">Y</span>
          <span><strong>Call Analyzer</strong><small>Administration</small></span>
        </div>
        <nav>
          {navigation.map((item) => {
            const active = pathname === item.href || (item.href === "/results" && pathname.startsWith("/calls/"));
            return <Link key={item.href} href={item.href} aria-current={active ? "page" : undefined} className={active ? "active" : ""}>{item.label}</Link>;
          })}
          {evaluationEnabled ? <Link href="/evaluation" className={pathname.startsWith("/evaluation") ? "active" : ""} aria-current={pathname.startsWith("/evaluation") ? "page" : undefined}>Evaluation</Link> : null}
        </nav>
      </aside>
      {menuOpen ? <button className="sidebar-scrim" type="button" aria-label="Close navigation" onClick={() => setMenuOpen(false)} /> : null}
      <div className="main-column">
        <header className="top-header">
          <button className="menu-button secondary button" type="button" onClick={() => setMenuOpen((open) => !open)} aria-expanded={menuOpen} aria-controls="main-navigation">Menu</button>
          <div>
            <h1>{pageTitle(pathname)}</h1>
            {pathname.startsWith("/processing/") && processingStatus ? <span className="header-context">{processingStatus}</span> : null}
          </div>
          <div className="header-account">
            <span title={user.email}>{user.email}</span>
            <button className="button secondary compact" type="button" onClick={() => void logout()} disabled={loggingOut}>{loggingOut ? "Signing out…" : "Sign out"}</button>
          </div>
        </header>
        <main id="main-content" className="page-content"><EvaluationAvailability value={evaluationEnabled}>{children}</EvaluationAvailability></main>
      </div>
    </div>
  );
}
