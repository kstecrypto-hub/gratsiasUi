type Props = {
  busy: boolean;
  disabled?: boolean;
  onTest: () => void;
};

export function YeastarTestConnectionButton({ busy, disabled = false, onTest }: Props) {
  return (
    <button
      className="button primary"
      type="button"
      disabled={disabled || busy}
      aria-busy={busy}
      onClick={onTest}
    >
      {busy ? "Testing connection..." : "Test connection"}
    </button>
  );
}
