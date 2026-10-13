---
name: wiki-doc-enhance
description: Enhance a Wiki concept doc (sources/*.md in this repo) by analyzing the SGLang / aiak_sglang codebase and adding hand-authored SVG diagrams. Organizes content as "logical what → logical abstraction → implementation (light on code)" and writes matching diagrams to docs/WikiImage. Use when the user asks to 增强/enhance a sources/*.md doc, explain an SGLang module in the Wiki, or add diagrams to a Wiki page.
---

# Wiki Doc Enhance

Enhance a concept doc under `sources/*.md` so a reader understands a system **conceptually first**, grounded in the real SGLang implementation, with diagrams that carry the explanation.

## When to use

The user asks to add or enhance a `sources/<topic>.md` page, to explain an SGLang module in the Wiki, or to add figures to a page. The companion codebase is the local SGLang fork, by default at `/Users/dingzifeng/Develop/aiak_sglang` (confirm the path if it is missing).

## The content organization (non-negotiable arc)

Every enhanced doc follows this order. This is the whole point of the skill — lead with meaning, end with code.

1. **逻辑侧：要做什么 (the problem).** Open with the problem in plain terms — what breaks, why the naive approach is insufficient. No code, no class names yet. The reader should understand *why this component needs to exist* before seeing how.
2. **逻辑抽象：应该怎么建立抽象 (the design).** Introduce the core concept/abstraction as an *idea*, not as a class. Explain the key insight that makes the design work (e.g. "comparing two integers", "separating addressing from storage"). Name types only after the idea lands.
3. **图片 (diagrams).** Add SVG diagrams at the points where a picture beats prose: the abstraction itself, a before/after or state-transition, a concrete worked example. See `references/svg-style.md`.
4. **结合实现，但克制 (implementation, light).** Ground the abstraction in the real code — correct file paths, class/function names, the one or two subtle engineering details worth knowing. Keep code snippets short and illustrative; prefer prose + diagram over pasting source. Cut anything that reads like a code dump.

Finish with a one-paragraph 小结 restating the core value of the abstraction.

## Workflow

1. **Read the current doc** fully (`sources/<topic>.md`) — know what's already there; you are reorganizing and deepening, not starting over.
2. **Locate the real code.** Grep the SGLang repo for the relevant module/classes. Verify every claim against source — class hierarchies, file paths, and APIs drift between versions. Delegate broad searches to the Explore/general-purpose agent to keep context lean; read the key files directly. Do **not** trust the old doc's class names or line numbers without re-checking.
3. **Restructure** the markdown into the four-part arc above. Trim excessive code; promote buried insights to the top.
4. **Author diagrams** as SVGs into `docs/WikiImage/` following `references/svg-style.md`. Reference them from the md with `![alt](../docs/WikiImage/<name>.svg)` (see conventions). Name files `<topic>_<aspect>.svg`.
5. **Validate** each SVG is well-formed XML, confirm no placeholder markers remain, and confirm the image references resolve.
6. **Report**: the HTML (`docs/<topic>.html`) is produced by an external build pipeline — do not hand-edit it. Tell the user to rebuild. Confirm the page is already listed in `sources/index.md` and `docs/.wiki_sidebar.html`; if not, add it.

See `references/wiki-conventions.md` for the repo's build/layout rules (image path rewriting, pandoc title, TOC behavior, sidebar).

## Guardrails

- **Accuracy over fluency.** A confident sentence about code you did not read is a bug. Re-verify names and paths in the current checkout.
- **Match the house style.** New diagrams must look like the existing `docs/WikiImage/*.svg` (same palette, white rounded panels, system font) — read `references/svg-style.md` before drawing.
- **Write in the doc's language** (these docs are Chinese; keep terminology consistent with sibling pages).
- **Chunk large writes.** SVGs and long md go in sequential Write/Edit calls, never one oversized tool call.
- Don't duplicate what the repo already records; add the non-obvious conceptual framing.
