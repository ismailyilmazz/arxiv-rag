import { useMemo } from "react";

import type { Plan } from "../api/types";
import { text } from "../strings";

interface Props {
  plan: Plan;
  removed: string[];
  collapsed: boolean;
  onToggle: (id: string) => void;
  onWrite: () => void;
}

export function PlanStep({ plan, removed, collapsed, onToggle, onWrite }: Props) {
  const scale = useMemo(() => {
    const values = plan.themes.flatMap((t) => t.papers.map((p) => p.relevance ?? 0));
    const low = Math.min(...values);
    const high = Math.max(...values);
    return (value: number | null) => (high > low ? 0.15 + (0.85 * ((value ?? low) - low)) / (high - low) : 1);
  }, [plan]);
  const kept = plan.screened - removed.length;

  return (
    <div className="plan">
      <p className="plan__heading">{text.planHeading(kept, plan.themes.length)}</p>
      {!collapsed && <p className="hint">{text.planHint}</p>}
      {plan.rejected_queries.length > 0 && (
        <p className="hint">
          {text.rejectedQueries} {plan.rejected_queries.join("; ")}
        </p>
      )}
      <details className="sources" open={!collapsed}>
        <summary className="sources__toggle">{text.showSources}</summary>
      {plan.themes.map((theme) => (
        <section key={theme.name} className="theme">
          <h3 className="theme__name">{theme.name}</h3>
          <ul className="papers">
            {theme.papers.map((paper) => {
              const included = !removed.includes(paper.id);
              return (
                <li key={paper.id} className={included ? "paper" : "paper paper--removed"}>
                  <input
                    type="checkbox"
                    aria-label={paper.title}
                    checked={included}
                    onChange={() => onToggle(paper.id)}
                  />
                  <div>
                    <a className="paper__title" href={paper.url} target="_blank" rel="noopener noreferrer">
                      {paper.title}
                    </a>
                    <p className="paper__meta">
                      {paper.id}
                      {paper.deep && <span className="paper__deep"> {text.deep}</span>}
                    </p>
                    <span
                      className="relevance"
                      role="img"
                      aria-label={`${text.relevance}: ${paper.relevance?.toFixed(3) ?? "?"}`}
                      style={{ inlineSize: `${Math.round(scale(paper.relevance) * 100)}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
      {plan.further_reading.length > 0 && (
        <section className="theme">
          <h3 className="theme__name">{text.furtherReading}</h3>
          <p className="hint">{text.furtherReadingHint}</p>
          <ul className="plain">
            {plan.further_reading.map((paper) => (
              <li key={paper.id}>
                <a href={paper.url} target="_blank" rel="noopener noreferrer">
                  {paper.title}
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}
      {plan.dropped_by_themes.length > 0 && (
        <details className="dropped">
          <summary>{text.dropped(plan.dropped_by_themes.length)}</summary>
          <ul className="plain">
            {plan.dropped_by_themes.map((paper) => (
              <li key={paper.id}>{paper.title}</li>
            ))}
          </ul>
        </details>
      )}
      </details>
      {!collapsed && kept === 0 && (
        <p className="field-error" role="alert">
          {text.noSourcesLeft}
        </p>
      )}
      {!collapsed && (
        <button className="button" type="button" onClick={() => kept > 0 && onWrite()}>
          {text.writeDocument}
        </button>
      )}
    </div>
  );
}
