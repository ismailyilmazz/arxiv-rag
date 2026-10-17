import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import type { Plan } from "../api/types";
import { text } from "../strings";
import { PlanStep } from "./PlanStep";

const plan: Plan = {
  plan_id: "p1",
  topic: "experts",
  doc_type: "survey",
  lang: "en",
  screened: 2,
  themes: [
    {
      name: "Routing",
      size: 2,
      papers: [
        { id: "2101.03961", title: "Switch Transformers", relevance: 0.95, deep: true, url: "https://arxiv.org/abs/2101.03961" },
        { id: "2401.04088", title: "Mixtral of Experts", relevance: 0.92, deep: false, url: "https://arxiv.org/abs/2401.04088" },
      ],
    },
  ],
  further_reading: [{ id: "2407.06204", title: "A Survey on MoE", url: "https://arxiv.org/abs/2407.06204" }],
  dropped_by_themes: [{ id: "2009.03300", title: "MMLU", relevance: 0.9 }],
  rejected_queries: [],
  forced: false,
};

function Harness({ onWrite }: { onWrite: (removed: string[]) => void }) {
  const [removed, setRemoved] = useState<string[]>([]);
  const toggle = (id: string) => setRemoved((r) => (r.includes(id) ? r.filter((x) => x !== id) : [...r, id]));
  return <PlanStep plan={plan} removed={removed} collapsed={false} onToggle={toggle} onWrite={() => onWrite(removed)} />;
}

describe("PlanStep", () => {
  it("shows themes, deep papers, further reading and dropped papers", () => {
    render(<Harness onWrite={() => {}} />);
    expect(screen.getByText(text.planHeading(2, 1))).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Routing" })).toBeInTheDocument();
    expect(screen.getAllByText(text.deep, { exact: false })).toHaveLength(1);
    expect(screen.getByRole("link", { name: "A Survey on MoE" })).toHaveAttribute("target", "_blank");
    expect(screen.getByText(text.dropped(1))).toBeInTheDocument();
  });

  it("removes sources and blocks writing when none are left", async () => {
    const onWrite = vi.fn();
    render(<Harness onWrite={onWrite} />);
    await userEvent.click(screen.getByRole("checkbox", { name: "Mixtral of Experts" }));
    expect(screen.getByText(text.planHeading(1, 1))).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: text.writeDocument }));
    expect(onWrite).toHaveBeenCalledWith(["2401.04088"]);
    await userEvent.click(screen.getByRole("checkbox", { name: "Switch Transformers" }));
    const button = screen.getByRole("button", { name: text.writeDocument });
    expect(button).toBeEnabled();
    expect(screen.getByRole("alert")).toHaveTextContent(text.noSourcesLeft);
    await userEvent.click(button);
    expect(onWrite).toHaveBeenCalledTimes(1);
  });
});
