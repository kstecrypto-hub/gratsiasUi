"use client";

import { createContext, useContext } from "react";

export const EvaluationAvailability = createContext(false);
export function EvaluationGate({ children }: { children: React.ReactNode }) {
  const enabled = useContext(EvaluationAvailability);
  if (!enabled) return <div className="notice" role="status">Evaluation is unavailable in this environment.</div>;
  return children;
}
