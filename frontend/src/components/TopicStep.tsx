import { useRef, useState } from "react";

import { api, messageOf } from "../api/client";
import type { DocType, Lang, PlanRequest, SearchResponse } from "../api/types";
import { docTypes, languages, text } from "../strings";

const MAX_SEEDS = 5;

interface Props {
  locked: boolean;
  onFound: (count: number) => void;
  onPlan: (request: PlanRequest) => void;
  onEdit: () => void;
}

export function TopicStep({ locked, onFound, onPlan, onEdit }: Props) {
  const [topic, setTopic] = useState("");
  const [docType, setDocType] = useState<DocType>("survey");
  const [lang, setLang] = useState<Lang>("tr");
  const [focus, setFocus] = useState("");
  const [found, setFound] = useState<SearchResponse | null>(null);
  const [seeds, setSeeds] = useState<string[]>([]);
  const [force, setForce] = useState(false);
  const [searching, setSearching] = useState(false);
  const [missingTopic, setMissingTopic] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  const search = async (event: React.FormEvent) => {
    event.preventDefault();
    if (searching) return;
    if (topic.trim().length < 2) {
      setMissingTopic(true);
      input.current?.focus();
      return;
    }
    setSearching(true);
    setError(null);
    try {
      const response = await api.search(topic.trim());
      setFound(response);
      setSeeds(response.results.slice(0, 3).map((r) => r.id));
      setForce(false);
      setNotice(null);
      if (response.results.length > 0) onFound(response.results.length);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setSearching(false);
    }
  };

  const submitOnEnter = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const toggleSeed = (id: string) => {
    if (seeds.includes(id)) {
      setSeeds(seeds.filter((s) => s !== id));
      setNotice(null);
    } else if (seeds.length >= MAX_SEEDS) {
      setNotice(text.seedLimit);
    } else {
      setSeeds([...seeds, id]);
      setNotice(null);
    }
  };

  const plan = () => {
    if (found && !found.accepted && !force) {
      setNotice(text.needForce);
      return;
    }
    setNotice(null);
    onPlan({ topic: topic.trim(), type: docType, lang, seeds, focus: focus.trim() || null, force });
  };

  if (locked) {
    return (
      <div className="summary">
        <p className="summary__topic">{topic}</p>
        <p className="summary__meta">
          {docTypes[docType].name}, {languages[lang]}
          {focus.trim() && `, ${focus.trim()}`}
        </p>
        <button className="link" type="button" onClick={onEdit}>
          {text.edit}
        </button>
      </div>
    );
  }

  return (
    <div className="topic">
      <form className="topic__form" onSubmit={search} noValidate>
        <label className="topic__question" htmlFor="topic">
          {text.topicQuestion}
        </label>
        <p className="topic__lead">{text.topicLead}</p>
        <textarea
          id="topic"
          ref={input}
          className="topic__input"
          rows={2}
          maxLength={300}
          placeholder={text.topicPlaceholder}
          value={topic}
          onChange={(e) => {
            setTopic(e.target.value);
            if (missingTopic && e.target.value.trim().length >= 2) setMissingTopic(false);
          }}
          onKeyDown={submitOnEnter}
          aria-invalid={missingTopic}
          aria-describedby={missingTopic ? "topic-error topic-hint" : "topic-hint"}
        />
        {missingTopic && (
          <p id="topic-error" className="field-error" role="alert">
            {text.topicMissing}
          </p>
        )}
        <p id="topic-hint" className="hint">
          {text.topicHint}
        </p>
        <fieldset className="choices">
          <legend>{text.documentType}</legend>
          {(Object.keys(docTypes) as DocType[]).map((key) => (
            <label key={key} className="choice">
              <input type="radio" name="doc-type" checked={docType === key} onChange={() => setDocType(key)} />
              <span className="choice__name">{docTypes[key].name}</span>
              <span className="choice__description">{docTypes[key].description}</span>
            </label>
          ))}
        </fieldset>
        <div className="row">
          <fieldset className="inline">
            <legend>{text.documentLanguage}</legend>
            {(Object.keys(languages) as Lang[]).map((key) => (
              <label key={key}>
                <input type="radio" name="lang" checked={lang === key} onChange={() => setLang(key)} /> {languages[key]}
              </label>
            ))}
          </fieldset>
          <div className="focus">
            <label htmlFor="focus">{text.focus}</label>
            <input
              id="focus"
              value={focus}
              maxLength={100}
              placeholder={text.focusPlaceholder}
              onChange={(e) => setFocus(e.target.value)}
              aria-describedby="focus-hint"
            />
            <p id="focus-hint" className="hint">
              {text.focusHint}
            </p>
          </div>
        </div>
        <button className="button" type="submit" aria-busy={searching}>
          {searching ? text.searching : text.search}
        </button>
      </form>

      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      {found && found.results.length === 0 && <p className="notice">{text.noResults}</p>}

      {found && found.results.length > 0 && (
        <fieldset className="results" id="results" tabIndex={-1}>
          <legend>{text.seedsLegend}</legend>
          <p className="results__count">{text.found(found.results.length, seeds.length)}</p>
          {found.translated_query && <p className="hint">{text.translatedQuery(found.translated_query)}</p>}
          <p className="hint">{text.seedsHint}</p>
          {!found.accepted && (
            <div className="notice">
              <p>{text.weakMatch}</p>
              <label>
                <input
                  type="checkbox"
                  checked={force}
                  onChange={(e) => {
                    setForce(e.target.checked);
                    if (e.target.checked) setNotice(null);
                  }}
                />{" "}
                {text.forceAnyway}
              </label>
            </div>
          )}
          <ul className="papers">
            {found.results.map((r) => (
              <li key={r.id} className={seeds.includes(r.id) ? "paper paper--picked" : "paper"}>
                <input
                  type="checkbox"
                  aria-label={r.title ?? r.id}
                  checked={seeds.includes(r.id)}
                  onChange={() => toggleSeed(r.id)}
                />
                <div>
                  <a className="paper__title" href={r.url} target="_blank" rel="noopener noreferrer">
                    {r.title ?? r.id}
                  </a>
                  {(r.authors || r.year || r.category) && (
                    <p className="paper__meta">
                      {[r.authors?.split(",").slice(0, 3).join(","), r.year, r.category].filter(Boolean).join(", ")}
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ul>
          {notice && (
            <p className="field-error" role="alert">
              {notice}
            </p>
          )}
          <button className="button" type="button" onClick={plan}>
            {text.createPlan}
          </button>
        </fieldset>
      )}
    </div>
  );
}
