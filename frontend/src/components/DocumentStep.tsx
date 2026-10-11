import { api } from "../api/client";
import type { Job } from "../api/types";
import { text } from "../strings";
import { Markdown } from "./Markdown";

export function DocumentStep({ job, markdown }: { job: Job; markdown: string }) {
  const words = Number(job.summary?.words ?? 0);
  const cited = Number(job.summary?.cited ?? 0);
  return (
    <div className="result">
      <div className="result__bar">
        <p className="result__summary">{text.documentSummary(words, cited)}</p>
        <a className="button" href={api.documentUrl(job.job_id)} download>
          {text.downloadMarkdown}
        </a>
        <a className="link" href={api.reportUrl(job.job_id)} download>
          {text.downloadReport}
        </a>
      </div>
      <Markdown source={markdown} />
    </div>
  );
}
