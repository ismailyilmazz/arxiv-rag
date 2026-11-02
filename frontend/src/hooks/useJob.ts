import { useEffect, useState } from "react";

import { ApiError, api, isTransient, messageOf } from "../api/client";
import type { Job } from "../api/types";
import { text } from "../strings";

export const MAX_RETRY_MS = 30000;

export function useJob(jobId: string | null, intervalMs: number) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    setJob(null);
    setError(null);
    setOffline(false);
    if (!jobId) return;
    let cancelled = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const later = (ms: number) => {
      clearTimeout(timer);
      timer = setTimeout(tick, ms);
    };
    const tick = async () => {
      try {
        const next = await api.getJob(jobId);
        if (cancelled) return;
        failures = 0;
        setOffline(false);
        setJob(next);
        if (next.status === "queued" || next.status === "running") later(intervalMs);
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) {
          setError(text.errors.jobLost);
        } else if (isTransient(e)) {
          failures += 1;
          setOffline(true);
          later(Math.min(intervalMs * 2 ** failures, MAX_RETRY_MS));
        } else {
          setError(messageOf(e));
        }
      }
    };
    const resume = () => {
      if (!cancelled) later(0);
    };
    window.addEventListener("online", resume);
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      window.removeEventListener("online", resume);
    };
  }, [jobId, intervalMs]);

  const active =
    jobId !== null && error === null && (job === null || job.status === "queued" || job.status === "running");
  return { job, error, active, offline };
}
