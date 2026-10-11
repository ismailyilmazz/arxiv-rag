import DOMPurify from "dompurify";
import { marked } from "marked";
import { useMemo } from "react";

const CITATION = /\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()/g;

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A" && /^https?:/.test(node.getAttribute("href") ?? "")) {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

export function renderMarkdown(source: string): string {
  const marked_up = source.replace(CITATION, '<span class="cite">[$1]</span>');
  return DOMPurify.sanitize(marked.parse(marked_up, { async: false }), { ADD_ATTR: ["target"] });
}

export function Markdown({ source }: { source: string }) {
  const html = useMemo(() => renderMarkdown(source), [source]);
  return <div className="document" dangerouslySetInnerHTML={{ __html: html }} />;
}
