# Design system

This document defines the visual and interaction system for the study app. Its
palette derives from the Kayzn logo — a geometric letter *K* of vertical bars —
applied through Streamlit's built-in theming.

## Principles

- **Evidence over decoration.** Every color, chart, and control carries
  meaning; nothing is styled for its own sake.
- **One accent.** Orange marks the decision model under study and interactive
  focus. It is never spent on incidental highlights.
- **Read-only calm.** The interface is a lens over cached results, so it favors
  quiet surfaces and stable layout over motion.
- **Native first.** Streamlit built-ins are preferred over third-party
  components; a dependency is added only when no built-in covers the need.
- **Legible in the dark.** Type, spacing, and contrast are tuned for a
  near-black canvas and verified against WCAG AA.

## Palette

The logo is warm off-white bars and three orange bars on a near-black field.
The app uses the same four families plus one muted neutral and one border
neutral. Ratios are measured against each color's own background.

| Role | Hex | Usage | Contrast |
| --- | --- | --- | --- |
| Background | `#0A0A0A` | App canvas | n/a (base surface) |
| Secondary background | `#161412` | Cards, widgets, sidebar, code blocks | 1.08:1 vs background (surface, not text) |
| Text | `#F1EDE4` | Body and headings | 16.95:1 on background |
| Primary (accent) | `#E0651F` | Links, focus, model-under-study | 5.70:1 on background; 5.29:1 on secondary |
| Muted neutral | `#9A9384` | Secondary text, captions, references | 6.49:1 on background |
| Border | `#6B6454` | Element and container borders | 3.37:1 on background |

Text and accent pairings clear the 4.5:1 body-text bar; the border clears the
3:1 non-text bar.

## Typography

The app uses the three fonts bundled with Streamlit, so no request ever leaves
for a third-party font host and the app stays privacy-clean.

- **Headings — Source Serif** (`headingFont = "serif"`): a serif gives section
  titles editorial weight and separates them from dense tabular body text.
- **Body and UI — Source Sans** (`font = "sans-serif"`): a humanist sans is the
  most legible choice for widget labels, tables, and metrics.
- **Code — Source Code** (`codeFont = "monospace"`): a fixed-width face keeps
  tool calls and JSON aligned and unambiguous.

Sizes and weights follow Streamlit defaults: base 16px / 400; `h1` 2.75rem /
700, `h2`-`h6` semibold 600; code 0.875rem / 400; metric values 2.25rem. The
serif is confined to headings; body and widget text stay in the readable sans.

## Layout

**Page header.** Every page opens with the same three-part header: an `st.title`
naming the page, one line of purpose beneath it, and — where a selection is
active — a breadcrumb of the current variant, task, and injection (for example
`baseline / task_042 / none`). The breadcrumb doubles as the shareable state.

**Sidebar.** The sidebar holds the logo, `st.navigation` page links, the version
and mode caption, and the footer. It carries no analysis controls; page-specific
selectors live on the page beside the data they filter.

**Spacing.** Separate major sections with subheaders rather than blank space.
Use `st.container(border=True)` for a self-contained unit (a verdict panel, a
metric row, an empty state) and plain containers for prose or a single table.

**Table formatting.** Percentages render to one decimal (`92.4%`); USD renders
with a `$` sign to four decimals (`$0.0123`); latency renders in milliseconds as
integers (`180 ms`). Apply these through `st.dataframe` `column_config`
(`NumberColumn` with `format`) so alignment and rounding are consistent.

## Components

Prefer Streamlit built-ins:

- `st.metric` (with `border=True`) for the overview and cascade numbers.
- `st.container(border=True)` to frame panels and empty states.
- `st.badge` for verdict outcomes (PASS / FAIL) and judge kinds.
- `st.pills` / `st.segmented_control` for the variant and injection selectors
  when the option count is small; `st.selectbox` when it is long.
- `st.dataframe` with `column_config` for every table.
- `st.page_link` for cross-page navigation and flip deep links.
- `st.dialog` to confirm a saved label; `st.status` while a gate recomputes.
- `st.tabs` for the ten gates; `st.expander` for tool results and long
  trajectories.

**No component library is added.** The built-ins above cover navigation,
selection, badges, and status; the surveyed libraries duplicate them or are not
maintained (see Decisions).

**Empty and loading states.** When a view has no data, state what will appear
and the command that produces it, in the pattern `What will appear here. Run:
<command>` — the sentence as a caption, the command as a `bash` code block. Wrap
on-demand recomputation in `st.status("Recomputing…")` so the wait is visible.

## Charts

**Altair** renders the app's interactive charts through `st.altair_chart`. It is
already a Streamlit dependency, so it adds no new package; it is theme-aware
(honoring `chartCategoricalColors`) and gives native tooltips. Meanwhile
`decision_judges/report.py` keeps Matplotlib for the static PNG and SVG the
README embeds. The thin-view allowlist gains exactly one name, `altair`.

Color assignment is fixed: **orange** `#E0651F` for the decision model under
study, **off-white** `#F1EDE4` for baselines, **muted neutral** `#9A9384` for
references and prior points. Charts favor the bar motif of the logo — vertical
bars for per-judge and per-gate comparisons — with lines reserved for
frontiers and reliability curves.

## Branding

The logo appears atop the sidebar via `st.logo` and as the favicon in
`st.set_page_config(page_icon=...)`. The sidebar footer reads **Built by Kayzn.
Open source under the Apache License 2.0.**, linking *Kayzn* to
[kayzn.io](https://kayzn.io) and the license to the repository `LICENSE`. Refer
to the owner as *Kayzn*; the app itself is *the study app*.

## Accessibility

| Pairing | Ratio | Bar |
| --- | --- | --- |
| Text on background | 16.95:1 | AA/AAA body |
| Accent on background | 5.70:1 | AA body |
| Muted text on background | 6.49:1 | AA body |
| Border on background | 3.37:1 | AA non-text |

Focus order follows source order: header, selectors, then results. Charts and
the logo carry text alternatives describing the finding, not the picture. Color
never carries meaning alone — verdicts pair color with a `PASS`/`FAIL` label,
matched steps pair the mark with `✓`/`✗`, and chart series are labeled in-legend
as well as colored.

## Configuration

Apply the theme by replacing the `[theme]` block in `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
backgroundColor = "#0A0A0A"
secondaryBackgroundColor = "#161412"
textColor = "#F1EDE4"
primaryColor = "#E0651F"
borderColor = "#6B6454"
showSidebarBorder = true
baseRadius = "small"
font = "sans-serif"
headingFont = "serif"
codeFont = "monospace"
chartCategoricalColors = ["#E0651F", "#F1EDE4", "#9A9384"]
```

## Decisions

| Decision | Alternative considered | Why |
| --- | --- | --- |
| Altair for interactive charts | Plotly | Altair ships with Streamlit and is theme-aware; Plotly is a large extra dependency for interactivity the app does not need. |
| Matplotlib for static exports | Altair everywhere | The README embeds committed PNG/SVG; Matplotlib produces stable files without a browser. |
| Bundled Source fonts | Google Fonts | Bundled fonts render identically offline and send no user data to a font host. |
| No component library | streamlit-extras | Actively maintained (1.3.0, 2026) but its helpers duplicate native `st.badge`, `st.metric`, and containers. |
| No component library | streamlit-shadcn-ui | Restyles widgets via a custom front-end runtime; cosmetic gain, real weight. |
| No component library | streamlit-antd-components | Ant Design menus and tags; `st.navigation`, `st.tabs`, and `st.pills` cover the need natively. |
| No component library | streamlit-elements | Last PyPI release 2022; effectively unmaintained and unnecessary for a read-only view. |
| No component library | streamlit-option-menu | Single custom nav menu (0.4.0, 2024); `st.navigation` already provides multipage navigation. |
| Dark base theme | Light base | The logo lives on near-black; a dark canvas keeps the app and its brand consistent. |

Sources: Streamlit theming reference
(<https://docs.streamlit.io/develop/api-reference/configuration/config.toml>),
theming and font guides
(<https://docs.streamlit.io/develop/concepts/configuration/theming>,
<https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts>),
and Nielsen Norman Group on empty states and progressive disclosure
(<https://www.nngroup.com/articles/progressive-disclosure/>).
