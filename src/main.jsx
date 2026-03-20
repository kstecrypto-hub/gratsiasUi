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
  const [isLoading, setIsLoading] = useState(false);

  const currentEndpoints = useMemo(() => WEBHOOKS[environment], [environment]);

  async function send(type) {
    const endpoint = currentEndpoints[type];
    setIsLoading(true);
    setStatus(`Sending ${type.toUpperCase()} using GET -> ${endpoint}`);

    try {
      const response = await fetch(endpoint, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      setStatus(`✅ ${type.toUpperCase()} sent successfully (HTTP ${response.status})`);
    } catch (error) {
      setStatus(`❌ ${type.toUpperCase()} failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="card">
        <h1>Webhook Sender</h1>
        <p className="subtitle">Windows browser → Docker backend at localhost:49078</p>

        <label htmlFor="env" className="label">Environment</label>
        <select
          id="env"
          className="select"
          value={environment}
          onChange={(event) => setEnvironment(event.target.value)}
          disabled={isLoading}
        >
          <option value="production">Production</option>
          <option value="test">Test</option>
        </select>

        <div className="buttons">
          <button type="button" onClick={() => send('email')} disabled={isLoading}>Send Email</button>
          <button type="button" onClick={() => send('sms')} disabled={isLoading}>Send SMS</button>
        </div>

        <div className="status" role="status" aria-live="polite">{status}</div>
        <div className="hint">
          Method: <code>GET</code><br />
          Email: <code>{currentEndpoints.email}</code><br />
          SMS: <code>{currentEndpoints.sms}</code>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
