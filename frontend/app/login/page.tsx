"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { ApiError, api, messageFromError } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("expired")) {
      setNotice("Your session expired. Sign in again to continue.");
    }
    let active = true;
    api.auth.me().then(() => {
      if (active) router.replace("/dashboard");
    }).catch(() => undefined);
    return () => { active = false; };
  }, [router]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await api.auth.login(email.trim(), password);
      router.replace("/dashboard");
      router.refresh();
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 429) {
        setError("Too many sign-in attempts. Wait a few minutes and try again.");
      } else if (caught instanceof ApiError && caught.status === 401) {
        setError("The email or password is incorrect.");
      } else {
        setError(messageFromError(caught, "Sign in is temporarily unavailable."));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-panel" aria-labelledby="login-title">
        <div className="login-brand"><span className="brand-mark" aria-hidden="true">Y</span><span>Yeastar Call Analyzer</span></div>
        <h1 id="login-title">Administrator sign in</h1>
        <p className="muted">Use the administrator account configured for this application.</p>
        {notice ? <div className="notice" role="status">{notice}</div> : null}
        {error ? <div className="form-error" role="alert">{error}</div> : null}
        <form onSubmit={submit} className="stack-form">
          <label>
            <span>Email</span>
            <input name="email" type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="username" required autoFocus />
          </label>
          <label>
            <span>Password</span>
            <input name="password" type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" required />
          </label>
          <button className="button primary full-width" type="submit" disabled={submitting}>{submitting ? "Signing in…" : "Sign in"}</button>
        </form>
      </section>
    </main>
  );
}
