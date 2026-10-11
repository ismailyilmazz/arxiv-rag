import { useEffect, useState } from "react";

import { api, messageOf } from "../api/client";
import type { Job } from "../api/types";

export function useJob(jobId: string | null, intervalMs: number) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setJob(null);
    setError(null);
    if (!jobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const next = await api.getJob(jobId);
        if (cancelled) return;
        setJob(next);
        if (next.status === "queued" || next.status === "running") timer = setTimeout(tick, intervalMs);
      } catch (e) {
        if (!cancelled) setError(messageOf(e));
      }
    };
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId, intervalMs]);

  const active =
    jobId !== null && error === null && (job === null || job.status === "queued" || job.status === "running");
  return { job, error, active };
}
