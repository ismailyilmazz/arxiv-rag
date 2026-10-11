import { describe, expect, it } from "vitest";

import { renderMarkdown } from "./Markdown";

describe("renderMarkdown", () => {
  it("removes scripts and event handlers", () => {
    const html = renderMarkdown('Text <script>alert(1)</script><img src=x onerror="alert(2)">');
    expect(html).not.toContain("<script");
    expect(html).not.toContain("onerror");
  });

  it("marks citations but leaves markdown links alone", () => {
    const html = renderMarkdown("Shown in [3] and [1, 4]. See [arXiv](https://arxiv.org/abs/2101.03961).");
    expect(html).toContain('<span class="cite">[3]</span>');
    expect(html).toContain('<span class="cite">[1, 4]</span>');
    expect(html).toContain('href="https://arxiv.org/abs/2101.03961"');
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
  });
});
