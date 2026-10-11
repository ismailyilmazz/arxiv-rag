import { useEffect } from "react";

import { text } from "../strings";

export interface Toast {
  id: number;
  group: "search" | "plan" | "document";
  message: string;
  action?: { label: string; target: string };
}

const LIFETIME_MS = 7000;

function reveal(target: string) {
  const element = document.getElementById(target);
  const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  element?.scrollIntoView?.({ behavior: reduce ? "auto" : "smooth", block: "start" });
  element?.focus({ preventScroll: true });
}

function ToastItem({ toast, onClose }: { toast: Toast; onClose: (id: number) => void }) {
  useEffect(() => {
    const timer = setTimeout(() => onClose(toast.id), LIFETIME_MS);
    return () => clearTimeout(timer);
  }, [toast.id, onClose]);

  return (
    <div className="toast">
      <p className="toast__message">{toast.message}</p>
      {toast.action && (
        <button
          className="toast__action"
          type="button"
          onClick={() => {
            reveal(toast.action!.target);
            onClose(toast.id);
          }}
        >
          {toast.action.label}
        </button>
      )}
      <button className="toast__close" type="button" aria-label={text.toasts.close} onClick={() => onClose(toast.id)}>
        <span aria-hidden="true">×</span>
      </button>
    </div>
  );
}

export function Toasts({ toasts, onClose }: { toasts: Toast[]; onClose: (id: number) => void }) {
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onClose={onClose} />
      ))}
    </div>
  );
}
