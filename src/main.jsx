import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const WEBHOOK_PATHS = {
  production: {
    email: 'webhook/email',
    sms: 'webhook/sms',
  },
  test: {
    email: 'webhook-test/email',
    sms: 'webhook-test/sms',
  },
};

function joinUrl(base, path) {
  const cleanedBase = base.replace(/\/+$/, '');
  return `${cleanedBase}/${path}`;
}

function App() {
  const [environment, setEnvironment] = useState('production');
  const [apiBaseUrl, setApiBaseUrl] = useState('http://localhost:49078');
  const [status, setStatus] = useState('Ready');
  const [isSending, setIsSending] = useState(false);

  const currentEndpoints = useMemo(() => {
    const paths = WEBHOOK_PATHS[environment];
    return {
      email: joinUrl(apiBaseUrl, paths.email),
      sms: joinUrl(apiBaseUrl, paths.sms),
    };
  }, [apiBaseUrl, environment]);

  async function pingBackend() {
    setIsSending(true);
    setStatus(`Checking backend at ${apiBaseUrl}...`);

    try {
      const response = await fetch(apiBaseUrl, { method: 'GET' });
      setStatus(`✅ Backend reachable (HTTP ${response.status})`);
    } catch (error) {
      setStatus(`❌ Backend is not reachable at ${apiBaseUrl}: ${error.message}`);
    } finally {
      setIsSending(false);
    }
  }

  async function triggerWebhook(type) {
    const endpoint = currentEndpoints[type];
    setIsSending(true);
    setStatus(`Sending ${type.toUpperCase()} to ${endpoint}...`);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source: 'gratsias-ui',
          channel: type,
          environment,
          timestamp: new Date().toISOString(),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      setStatus(`✅ ${type.toUpperCase()} webhook sent successfully (${response.status})`);
    } catch (error) {
      setStatus(`❌ Failed to send ${type.toUpperCase()} webhook at ${endpoint}: ${error.message}`);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="page">
      <section className="card">
        <h1>Webhook Sender</h1>
        <p className="subtitle">Dark mode control panel for Email and SMS triggers.</p>

        <label htmlFor="base" className="label">Backend Base URL</label>
        <div className="row">
          <input
            id="base"
            className="input"
            type="url"
            value={apiBaseUrl}
            onChange={(event) => setApiBaseUrl(event.target.value)}
            placeholder="http://localhost:49078"
            disabled={isSending}
          />
          <button type="button" onClick={pingBackend} disabled={isSending}>Check</button>
        </div>

        <label htmlFor="env" className="label">Environment</label>
        <select
          id="env"
          className="select"
          value={environment}
          onChange={(event) => setEnvironment(event.target.value)}
          disabled={isSending}
        >
          <option value="production">Production</option>
          <option value="test">Test</option>
        </select>

        <div className="buttons">
          <button type="button" onClick={() => triggerWebhook('email')} disabled={isSending}>Send Email</button>
          <button type="button" onClick={() => triggerWebhook('sms')} disabled={isSending}>Send SMS</button>
        </div>

        <div className="status" role="status" aria-live="polite">{status}</div>
        <div className="hint">
          Current endpoints: <code>{currentEndpoints.email}</code> and <code>{currentEndpoints.sms}</code>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
