import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { docTypes, text } from "./strings";

const results = ["2101.03961", "1701.06538", "2401.04088", "2202.08906"].map((id, i) => ({
  id,
  url: `https://arxiv.org/abs/${id}`,
  title: `Paper ${i}`,
  authors: "A, B",
  year: "2021",
  category: "cs.LG",
  score: 1 - i / 10,
}));

const plan = {
  plan_id: "p1",
  topic: "mixture of experts",
  doc_type: "proposal",
  lang: "tr",
  screened: 2,
  themes: [
    {
      name: "Routing",
      size: 2,
      papers: results.slice(0, 2).map((r) => ({ id: r.id, title: r.title, relevance: 0.95, deep: false, url: r.url })),
    },
  ],
  further_reading: [],
  dropped_by_themes: [],
  rejected_queries: [],
  forced: false,
};

function fakeServer() {
  const calls: { url: string; body?: unknown }[] = [];
  const polls: Record<string, number> = {};
  const job = (id: string, kind: string, extra: object) => ({
    job_id: id, kind, stage: "sections", done: 1, total: 3, position: 0, error: null, plan_id: null, summary: null, ...extra,
  });
  const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });
  return {
    calls,
    fetch: vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url === "/api/search") return json({ query: "q", accepted: true, guard_prob: 0.9, results });
      if (url === "/api/plans") return json({ job_id: "plan-job", position: 1 });
      if (url === "/api/documents") return json({ job_id: "doc-job", position: 1 });
      if (url === "/api/plans/p1") return json(plan);
      if (url.startsWith("/api/jobs/") && !url.endsWith("/document")) {
        const id = url.split("/")[3];
        polls[id] = (polls[id] ?? 0) + 1;
        const finished = polls[id] > 1;
        if (id === "plan-job")
          return json(job(id, "plan", finished ? { status: "done", stage: "done", plan_id: "p1" } : { status: "running" }));
        return json(job(id, "document", finished
          ? { status: "done", stage: "done", plan_id: "p1", summary: { words: 1234, cited: 2 } }
          : { status: "running" }));
      }
      if (url === "/api/jobs/doc-job/document") return new Response("# Başlık\n\nBulgu [1].\n", { status: 200 });
      return new Response(JSON.stringify({ detail: "yok" }), { status: 404 });
    }),
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("goes from a topic to a written document", async () => {
    const server = fakeServer();
    vi.stubGlobal("fetch", server.fetch);
    const user = userEvent.setup();
    render(<App pollMs={5} />);

    await user.type(screen.getByLabelText(text.topicQuestion), "mixture of experts");
    await user.click(screen.getByLabelText(new RegExp(docTypes.proposal.name)));
    await user.type(screen.getByLabelText(text.focus), "routing");
    await user.click(screen.getByRole("button", { name: text.search }));
    expect(await screen.findByText(text.toasts.found(4))).toBeInTheDocument();
    expect(screen.getByText(text.found(4, 3))).toBeInTheDocument();
    const boxes = screen.getAllByRole("checkbox");
    expect(boxes.filter((b) => (b as HTMLInputElement).checked)).toHaveLength(3);
    await user.click(screen.getByRole("button", { name: text.toasts.close }));
    expect(screen.queryByText(text.toasts.found(4))).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: text.createPlan }));
    expect(await screen.findByText(text.toasts.planAccepted)).toBeInTheDocument();
    expect(await screen.findByText(text.planHeading(2, 1))).toBeInTheDocument();
    expect(screen.getByText(text.toasts.planReady)).toBeInTheDocument();
    expect(screen.queryByText(text.toasts.planAccepted)).not.toBeInTheDocument();
    expect(server.calls.find((c) => c.url === "/api/plans")?.body).toEqual({
      topic: "mixture of experts", type: "proposal", lang: "tr",
      seeds: ["2101.03961", "1701.06538", "2401.04088"], focus: "routing", force: false,
    });

    const planList = screen.getByRole("heading", { name: "Routing" }).parentElement!;
    await user.click(planList.querySelectorAll("input")[1]);
    expect(screen.getByText("mixture of experts", { selector: ".summary__topic" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: text.search })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: text.writeDocument }));
    expect(screen.queryByRole("button", { name: text.writeDocument })).not.toBeInTheDocument();
    expect(screen.getByText(text.showSources)).toBeInTheDocument();

    expect(await screen.findByRole("heading", { name: "Başlık" })).toBeInTheDocument();
    expect(screen.getByText(text.toasts.documentReady)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: text.toasts.goDocument })).toBeInTheDocument();
    expect(screen.getByText(text.documentSummary(1234, 2))).toBeInTheDocument();
    expect(screen.getByRole("link", { name: text.downloadMarkdown })).toHaveAttribute("href", "/api/jobs/doc-job/document");
    expect(server.calls.find((c) => c.url === "/api/documents")?.body).toEqual({
      plan_id: "p1", removed_ids: ["1701.06538"],
    });
    await waitFor(() => expect(screen.getByRole("button", { name: text.edit })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: text.edit }));
    expect(screen.getByLabelText(text.topicQuestion)).toHaveValue("mixture of experts");
    expect(screen.getByLabelText(new RegExp(docTypes.proposal.name))).toBeChecked();
    expect(screen.queryByRole("heading", { name: text.steps.plan })).not.toBeInTheDocument();
  });

  it("asks before planning a topic the guard rejected", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({ query: "pizza", accepted: false, guard_prob: 0.1, results }), { status: 200 })));
    const user = userEvent.setup();
    render(<App pollMs={5} />);
    await user.type(screen.getByLabelText(text.topicQuestion), "pizza tarifi{Enter}");
    expect(await screen.findByText(text.weakMatch)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: text.createPlan })).toBeDisabled();
    await user.click(screen.getByLabelText(text.forceAnyway));
    expect(screen.getByRole("button", { name: text.createPlan })).toBeEnabled();
  });

  it("explains an empty topic instead of silently blocking the search", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App pollMs={5} />);
    const button = screen.getByRole("button", { name: text.search });
    expect(button).toBeEnabled();
    await user.click(button);
    expect(screen.getByRole("alert")).toHaveTextContent(text.topicMissing);
    expect(screen.getByLabelText(text.topicQuestion)).toHaveFocus();
    expect(fetchMock).not.toHaveBeenCalled();
    await user.type(screen.getByLabelText(text.topicQuestion), "rag");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
