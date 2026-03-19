import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const WEBHOOKS = {
  production: {
    email: 'http://localhost:49078/webhook/email',
    sms: 'http://localhost:49078/webhook/sms',
  },
  test: {
    email: 'http://localhost:49078/webhook-test/email',
    sms: 'http://localhost:49078/webhook-test/sms',
  },
};

function App() {
  const [environment, setEnvironment] = useState('production');
  const [status, setStatus] = useState('Ready');
  const [isSending, setIsSending] = useState(false);

  const currentEndpoints = useMemo(() => WEBHOOKS[environment], [environment]);

  async function triggerWebhook(type) {
    const endpoint = currentEndpoints[type];
    setIsSending(true);
    setStatus(`Sending ${type.toUpperCase()} to ${environment}...`);

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
      setStatus(`❌ Failed to send ${type.toUpperCase()} webhook: ${error.message}`);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="page">
      <section className="card">
        <h1>Webhook Sender</h1>
        <p className="subtitle">Dark mode control panel for Email and SMS triggers.</p>

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
          <button
            type="button"
            onClick={() => triggerWebhook('email')}
            disabled={isSending}
          >
            Send Email
          </button>
          <button
            type="button"
            onClick={() => triggerWebhook('sms')}
            disabled={isSending}
          >
            Send SMS
          </button>
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
