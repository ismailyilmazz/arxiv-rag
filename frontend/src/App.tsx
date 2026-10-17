import { useCallback, useEffect, useRef, useState } from "react";

import { api, messageOf } from "./api/client";
import type { Plan, PlanRequest } from "./api/types";
import { DocumentStep } from "./components/DocumentStep";
import { JobStatus } from "./components/JobStatus";
import { PlanStep } from "./components/PlanStep";
import type { Toast } from "./components/Toasts";
import { Toasts } from "./components/Toasts";
import { TopicStep } from "./components/TopicStep";
import { useJob } from "./hooks/useJob";
import { text } from "./strings";

function StepTitle({ id, children }: { id: string; children: string }) {
  const ref = useRef<HTMLHeadingElement>(null);
  useEffect(() => ref.current?.focus(), []);
  return (
    <h2 className="step__title" id={id} ref={ref} tabIndex={-1}>
      {children}
    </h2>
  );
}

function Progress({ current, finished }: { current: number; finished: boolean }) {
  const names = [text.steps.topic, text.steps.plan, text.steps.document];
  return (
    <ol className="progress" aria-label={text.progress}>
      {names.map((name, index) => {
        const step = index + 1;
        const state = step < current || finished ? "done" : step === current ? "current" : "next";
        return (
          <li key={name} className={`progress__step progress__step--${state}`}
              aria-current={state === "current" ? "step" : undefined}>
            {name}
          </li>
        );
      })}
    </ol>
  );
}

export function App({ pollMs = 1500 }: { pollMs?: number }) {
  const [round, setRound] = useState(0);
  const [planJobId, setPlanJobId] = useState<string | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [removed, setRemoved] = useState<string[]>([]);
  const [docJobId, setDocJobId] = useState<string | null>(null);
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const lastToast = useRef(0);
  const sending = useRef(false);
  const planJob = useJob(planJobId, pollMs);
  const docJob = useJob(docJobId, pollMs);
  const busy = planJob.active || docJob.active;

  const notify = useCallback((group: Toast["group"], message: string, action?: Toast["action"]) => {
    lastToast.current += 1;
    const toast = { id: lastToast.current, group, message, action };
    setToasts((current) => [...current.filter((t) => t.group !== group), toast]);
  }, []);
  const closeToast = useCallback((id: number) => setToasts((current) => current.filter((t) => t.id !== id)), []);

  const planId = planJob.job?.status === "done" ? planJob.job.plan_id : null;
  useEffect(() => {
    if (!planId) return;
    api
      .getPlan(planId)
      .then((loaded) => {
        setPlan(loaded);
        notify("plan", text.toasts.planReady, { label: text.toasts.goPlan, target: "step-plan" });
      })
      .catch((e) => setError(messageOf(e)));
  }, [planId, notify]);

  const finishedDocument = docJob.job?.status === "done" ? docJob.job.job_id : null;
  useEffect(() => {
    if (!finishedDocument) return;
    api
      .documentText(finishedDocument)
      .then((loaded) => {
        setMarkdown(loaded);
        notify("document", text.toasts.documentReady, { label: text.toasts.goDocument, target: "step-document" });
      })
      .catch((e) => setError(messageOf(e)));
  }, [finishedDocument, notify]);

  const found = useCallback(
    (count: number) => notify("search", text.toasts.found(count), { label: text.toasts.goResults, target: "results" }),
    [notify],
  );

  const clearJobs = () => {
    setPlanJobId(null);
    setPlan(null);
    setRemoved([]);
    setDocJobId(null);
    setMarkdown(null);
    setError(null);
  };

  const once = async (action: () => Promise<void>) => {
    if (sending.current) return;
    sending.current = true;
    try {
      await action();
    } finally {
      sending.current = false;
    }
  };

  const startPlan = (request: PlanRequest) =>
    once(async () => {
      clearJobs();
      try {
        setPlanJobId((await api.createPlan(request)).job_id);
        notify("plan", text.toasts.planAccepted);
      } catch (e) {
        setError(messageOf(e));
      }
    });

  const write = () =>
    once(async () => {
      if (!plan) return;
      setError(null);
      setMarkdown(null);
      try {
        setDocJobId((await api.createDocument(plan.plan_id, removed)).job_id);
        notify("document", text.toasts.documentAccepted);
      } catch (e) {
        setError(messageOf(e));
      }
    });

  const whenIdle = (action: () => void) => () => (busy ? notify("plan", text.waitForJob) : action());

  const toggle = (id: string) =>
    setRemoved((current) => (current.includes(id) ? current.filter((r) => r !== id) : [...current, id]));

  const reset = () => {
    setRound((r) => r + 1);
    clearJobs();
  };

  const current = docJobId ? 3 : planJobId ? 2 : 1;
  return (
    <>
      <header className="masthead">
        <p className="masthead__name">{text.title}</p>
        <Progress current={current} finished={markdown !== null} />
        {planJobId && (
          <button className="link masthead__reset" type="button" onClick={whenIdle(reset)}>
            {text.newTopic}
          </button>
        )}
      </header>
      <main className="page">
        <ol className="steps">
          <li className="step">
            <h2 className="step__title">{text.steps.topic}</h2>
            <TopicStep
              key={round}
              locked={planJobId !== null}
              onFound={found}
              onPlan={startPlan}
              onEdit={whenIdle(clearJobs)}
            />
          </li>
          {planJobId && (
            <li className="step">
              <StepTitle id="step-plan">{text.steps.plan}</StepTitle>
              <JobStatus job={planJob.job} error={planJob.error} />
              {plan && (
                <PlanStep
                  plan={plan}
                  removed={removed}
                  collapsed={docJobId !== null && docJob.job?.status !== "failed"}
                  onToggle={toggle}
                  onWrite={write}
                />
              )}
            </li>
          )}
          {docJobId && (
            <li className="step">
              <StepTitle id="step-document">{text.steps.document}</StepTitle>
              <JobStatus job={docJob.job} error={docJob.error} />
              {markdown !== null && docJob.job && <DocumentStep job={docJob.job} markdown={markdown} />}
            </li>
          )}
        </ol>
        {error && (
          <p className="notice notice--error" role="alert">
            {error}
          </p>
        )}
      </main>
      <footer className="footer">
        <p>{text.footer}</p>
      </footer>
      <Toasts toasts={toasts} onClose={closeToast} />
    </>
  );
}
