import { text } from "../strings";
import type { Job, JobCreated, Plan, PlanRequest, SearchResponse } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, text.errors.network);
  }
  if (!response.ok) {
    let message = text.errors.unknown;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = text.errors.invalid;
    } catch {
      message = text.errors.unknown;
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export const api = {
  search: (query: string, k = 10) =>
    request<SearchResponse>("/search", { method: "POST", body: JSON.stringify({ query, k }) }),
  createPlan: (body: PlanRequest) => request<JobCreated>("/plans", { method: "POST", body: JSON.stringify(body) }),
  getPlan: (planId: string) => request<Plan>(`/plans/${planId}`),
  createDocument: (planId: string, removedIds: string[]) =>
    request<JobCreated>("/documents", {
      method: "POST",
      body: JSON.stringify({ plan_id: planId, removed_ids: removedIds }),
    }),
  getJob: (jobId: string) => request<Job>(`/jobs/${jobId}`),
  documentUrl: (jobId: string) => `/api/jobs/${jobId}/document`,
  reportUrl: (jobId: string) => `/api/jobs/${jobId}/report`,
  async documentText(jobId: string): Promise<string> {
    const response = await fetch(api.documentUrl(jobId));
    if (!response.ok) throw new ApiError(response.status, text.errors.unknown);
    return response.text();
  },
};

export function messageOf(error: unknown): string {
  return error instanceof ApiError ? error.message : text.errors.unknown;
}
