# SVG diagram style guide

Diagrams are hand-authored SVG written directly to `docs/WikiImage/`. They must read as one visual system with the existing figures. Do not use a chart library or drawio for these — write the SVG by hand so you control the layout and it matches.

## Canvas

- Root: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 W H" font-family="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif">`. Set `viewBox` only (no fixed width/height) so the figure scales responsively in the page.
- First child is a full-canvas white panel: `<rect x="0" y="0" width="W" height="H" rx="12" fill="#ffffff"/>`. Diagrams render on light theme — never rely on a transparent background.
- Width ~820–920 for wide flows; height to fit. Title at `y≈34` (font-size 20, weight 700, `#212529`), one-line subtitle at `y≈56` (font-size 13, `#868e96`).

## Palette (brand-neutral, consistent across all figures)

Use these fixed roles. Each "family" = a saturated stroke/text color + a pale fill.

| Role | Stroke / text | Pale fill | Use for |
|------|---------------|-----------|---------|
| Blue | `#4c6ef5` / `#3b5bdb` | `#edf2ff` | primary layer, "attention"/"input" state |
| Teal | `#0ca678` / `#0b7285` | `#e6fcf5` | second layer, "full"/"storage" state |
| Amber | `#f08c00` / `#e8590c` | `#fff9db` / `#fff4e6` | third layer, highlights, communication points |
| Grape | `#7048e8` / `#5f3dc4` | `#f3f0ff` / `#e5dbff` | "scattered"/alternate state |
| Red | `#fa5252` / `#c92a2a` | `#fff5f5` | evictable / warning |
| Neutral | `#adb5bd` / `#495057` | `#f1f3f5` / `#f8f9fa` | generic boxes, notes, chrome |

Text colors: headings `#212529`, body `#495057`, secondary/caption `#868e96`. Arrows/connectors `#868e96`, 2px.

## Building blocks

- **Box**: `<rect rx="8..10">` with a pale fill + 1px saturated stroke of the same family. Title line font-size 12.5–14 weight 700 in the stroke color; body lines 11–12 in `#495057`/`#868e96`.
- **Layer banner** (for stacked architecture diagrams): a solid-color header strip on top of a pale box — draw the full box first, then a `rx` rect of header height in the saturated color, then white title text over it.
- **Arrow**: `<path ... marker-end="url(#ar)">` with a `<marker>` in `<defs>`:
  ```xml
  <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">
    <path d="M0 0 L8 4 L0 8 z" fill="#868e96"/>
  </marker>
  ```
  Use a distinct marker id per color if you need colored arrows.
- **Monospace data** (indices, shapes, token ids): `font-family="monospace"`, colored by the state it belongs to. Great for making abstractions concrete with a worked example.
- **Note strip**: a `#f8f9fa` box with `#e9ecef` stroke at the bottom holding the "key idea / takeaway" in 12px text. Most figures end with one.
- **Bracket**: a 3-segment `<path fill="none" stroke=... >` above a span of cells to group/annotate them.

## Composition patterns that work here

- **Stacked layers + right-side example**: wide layer boxes down the left ~65%, a narrow "concrete values" column on the right showing one worked example per layer (see `memory_pool_overview.svg`).
- **Horizontal pipeline**: 3–4 stage boxes left→right joined by labelled arrows; a parallel dashed "side lane" below for a secondary flow (see `communicator_layer_stages.svg`).
- **State grid**: rows = modes/states, columns = ranks/entities, each cell a colored box with the held value; brackets above group columns (see `communicator_scatter_modes.svg`).
- **Vertical trace**: a spine line with step dots; step boxes on the left, resulting shape/state on the right; highlight the 1–2 steps that actually matter in amber (see `communicator_dp_attn_flow.svg`).

## Rules

- Keep it legible: generous padding, don't crowd text to box edges, align on a grid.
- Pick 2–4 color families per figure, by semantic role — not decoration.
- Label the *insight*, not just the parts: every figure should make one idea click.
- Validate before finishing: `python3 -c "import xml.dom.minidom; xml.dom.minidom.parse('docs/WikiImage/<f>.svg')"`.
- Write large SVGs in chunks (initial Write ≤ ~50 lines, then Edit-append) rather than one huge tool call.
