export type DocType = "survey" | "proposal" | "synthesis";
export type Lang = "en" | "tr";
export type JobStatus = "queued" | "running" | "done" | "failed";
export type JobStage = "sources" | "cards" | "sections" | "translate" | "verify" | "done";

export interface SearchResult {
  id: string;
  url: string;
  title?: string;
  authors?: string;
  year?: string | null;
  category?: string | null;
  score: number;
}

export interface SearchResponse {
  query: string;
  accepted: boolean;
  guard_prob: number | null;
  results: SearchResult[];
}

export interface PlanRequest {
  topic: string;
  type: DocType;
  lang: Lang;
  seeds: string[];
  focus: string | null;
  force: boolean;
}

export interface JobCreated {
  job_id: string;
  position: number;
}

export interface Job {
  job_id: string;
  kind: "plan" | "document";
  status: JobStatus;
  stage: JobStage | null;
  done: number | null;
  total: number | null;
  position: number;
  error: string | null;
  plan_id: string | null;
  summary: Record<string, string | number | boolean | null> | null;
}

export interface PlanPaper {
  id: string;
  title: string;
  relevance: number | null;
  deep: boolean;
  url: string;
}

export interface PlanTheme {
  name: string;
  size: number;
  papers: PlanPaper[];
}

export interface Plan {
  plan_id: string;
  topic: string;
  doc_type: DocType;
  lang: Lang;
  screened: number;
  themes: PlanTheme[];
  further_reading: { id: string; title: string; url: string }[];
  dropped_by_themes: { id: string; title: string; relevance: number | null }[];
  rejected_queries: string[];
  forced: boolean;
}
