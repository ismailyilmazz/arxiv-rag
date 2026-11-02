import type { Job } from "../api/types";
import { stages, text } from "../strings";

export function JobStatus({ job, error, offline = false }: { job: Job | null; error: string | null; offline?: boolean }) {
  if (error) return <p className="notice notice--error" role="alert">{error}</p>;
  if (offline) return <p className="notice" role="status">{text.offline}</p>;
  if (job?.status === "failed") return <p className="notice notice--error" role="alert">{job.error}</p>;
  if (job?.status === "done") return null;
  const label = !job
    ? text.waiting
    : job.status === "queued"
      ? text.queued(job.position)
      : job.stage
        ? stages[job.stage]
        : text.waiting;
  const counted = job?.status === "running" && job.total ? job.total : 0;
  return (
    <div className="job" aria-live="polite">
      <p className="job__label">
        {label}
        {counted > 0 && <span className="job__count"> {job?.done} / {counted}</span>}
      </p>
      <progress className="job__bar" max={counted || undefined} value={counted ? (job?.done ?? 0) : undefined} />
    </div>
  );
}
