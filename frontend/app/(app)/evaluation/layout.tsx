import { EvaluationGate } from "@/components/evaluation-availability";

export default function EvaluationLayout({ children }: { children: React.ReactNode }) {
  return <EvaluationGate>{children}</EvaluationGate>;
}
