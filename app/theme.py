"""CRM-Adv design system: CSS, cards, charts.

The palette is defined once in ``src/config.py`` and imported here, so the
Streamlit pages and the saved figures cannot drift apart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import streamlit as st

from src.config import PALETTE as C

SERIES = [C["purple_deep"], C["lavender"], C["orange"], C["blue"],
          C["green"], C["pink"]]
MODEL_COLORS = {
    "KNN": C["lavender"],
    "GaussianNB": C["orange"],
    "LinearSVM": C["blue"],
    "RBFSVM": C["purple_deep"],
}
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "marketmind", ["#F5F2FF", "#CFC6FF", C["purple_deep"]]
)

plt.rcParams.update({
    "figure.facecolor": C["workspace"],
    "axes.facecolor": C["workspace"],
    "savefig.facecolor": C["workspace"],
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "text.color": C["text"],
    "axes.labelcolor": C["muted"],
    "axes.edgecolor": C["border"],
    "axes.linewidth": 0.8,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.titlecolor": C["text"],
    "xtick.color": C["muted"],
    "ytick.color": C["muted"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
})

ICON_PATHS = {
    "database": '<ellipse cx="12" cy="5.5" rx="7.5" ry="3"/><path d="M4.5 5.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/><path d="M4.5 11.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/>',
    "history": '<circle cx="12" cy="12" r="8"/><path d="M12 7.5V12l3 2"/>',
    "chart": '<path d="M5 19V10"/><path d="M10 19V5"/><path d="M15 19v-6"/><path d="M20 19V8"/>',
    "users": '<circle cx="9" cy="8" r="3"/><path d="M3.5 19c.5-3.3 2.6-5 5.5-5s5 1.7 5.5 5"/><circle cx="17" cy="9" r="2.4"/><path d="M15.5 14.4c2.6-.2 4.4 1.3 5 4.6"/>',
    "spark": '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z"/>',
    "clock": '<circle cx="12" cy="12" r="8"/><path d="M12 8v4.5l3 1.8"/>',
    "check": '<path d="M5 12.5l4.2 4.2L19 7.5"/>',
    "bolt": '<path d="M13 3L5 14h5l-1 7 8-11h-5l1-7Z"/>',
    "table": '<rect x="4" y="5" width="16" height="14" rx="2.5"/><path d="M4 10h16"/><path d="M9 10v9"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1.2l2-1.5-2-3.4-2.3 1a7 7 0 0 0-2-1.2L14.2 3h-4.4l-.4 2.7a7 7 0 0 0-2 1.2l-2.3-1-2 3.4 2 1.5A7 7 0 0 0 5 12c0 .4 0 .8.1 1.2l-2 1.5 2 3.4 2.3-1a7 7 0 0 0 2 1.2l.4 2.7h4.4l.4-2.7a7 7 0 0 0 2-1.2l2.3 1 2-3.4-2-1.5c.1-.4.1-.8.1-1.2Z"/>',
    "info": '<circle cx="12" cy="12" r="8"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
    "layers": '<path d="M12 4l8 4-8 4-8-4 8-4Z"/><path d="M4 12l8 4 8-4"/><path d="M4 16l8 4 8-4"/>',
    "arrow_right": '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
}


def icon(name, size=16, stroke=1.8):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICON_PATHS.get(name, ICON_PATHS["spark"])}</svg>')


CSS = f"""
<style>
:root {{
  --mm-purple: {C['purple']};
  --mm-purple-deep: {C['purple_deep']};
  --mm-lavender: {C['lavender']};
  --mm-border: {C['border']};
  --mm-muted: {C['muted']};
  --mm-text: {C['text']};
}}
html, body, [class*="css"] {{
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
               "Helvetica Neue", Arial, sans-serif;
  color: {C['text']};
}}
.stApp {{ background: {C['page']}; }}
header[data-testid="stHeader"] {{ background: {C['page']}; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ max-width: 1280px; padding: 1.4rem 1.8rem 3rem; }}

/* Streamlit 1.50 collapse: a stretched chart inside a bordered column reports
   the column gap (16px) as its width. Force image wrappers to fill the card. */
[data-testid="stImage"], [data-testid="stImageContainer"] {{ width: 100% !important; }}
[data-testid="stImageContainer"] img {{ width: 100% !important; height: auto !important; }}

/* Five tabs on a 768px viewport: trim the tab padding instead of scrolling. */
[data-testid="stTabs"] button[role="tab"] {{ padding-left: 0.5rem; padding-right: 0.5rem; }}

/* A donut is a small visual: cap it instead of stretching it full width when
   its column stacks. */
.st-key-donut_dashboard [data-testid="stImage"],
.st-key-donut_dashboard [data-testid="stImageContainer"],
.st-key-donut_dataset [data-testid="stImage"],
.st-key-donut_dataset [data-testid="stImageContainer"] {{ max-width: 420px !important; }}

/* Narrow viewports: columns stack so charts and forms keep a usable width. */
@media (max-width: 900px) {{
  [data-testid="stHorizontalBlock"] {{ flex-direction: column; gap: 0.9rem; }}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: 100% !important; flex: 1 1 100% !important;
  }}
}}

/* On tablet widths a 1/3 chart column is too narrow; stack only rows that
   contain a figure, leaving form and KPI columns side by side. */
@media (max-width: 1100px) {{
  [data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) {{
    flex-direction: column; gap: 0.9rem;
  }}
  [data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] {{
    width: 100% !important; flex: 1 1 100% !important;
  }}
}}

/* ---------------------------------------------------------------- sidebar */
section[data-testid="stSidebar"] {{
  background: {C['sidebar']};
  border-right: 1px solid {C['border']};
  min-width: 248px;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0.6rem; }}
section[data-testid="stSidebar"] .block-container {{ padding: 0; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.15rem; }}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding: 0.9rem 0.9rem 1.2rem; }}

.mm-brand {{ display: flex; align-items: center; gap: 10px; padding: 2px 6px 10px; }}
.mm-brand-mark {{
  width: 36px; height: 36px; border-radius: 11px; flex: 0 0 36px;
  background: linear-gradient(135deg, {C['gradient_from']}, {C['gradient_to']});
  display: flex; align-items: center; justify-content: center; color: white;
  box-shadow: 0 4px 10px rgba(121, 105, 232, 0.24);
}}
.mm-brand-name {{ font-size: 15px; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2; }}
.mm-brand-sub {{ font-size: 10px; color: {C['muted']}; margin-top: 1px; }}
.mm-nav-section {{
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.13em; color: #A0A3AC;
  margin: 14px 8px 3px; text-transform: uppercase;
}}
section[data-testid="stSidebar"] .stButton > button {{
  justify-content: flex-start; width: 100%; border: 1px solid transparent;
  background: transparent; color: #4B4F58; font-weight: 500; font-size: 12.5px;
  border-radius: 9px; padding: 0.42rem 0.65rem; min-height: 2.1rem;
  box-shadow: none; transition: background 120ms ease, color 120ms ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: {C['hover']}; color: {C['purple_deep']}; border-color: transparent;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: {C['lavender']}; color: {C['purple_deep']}; font-weight: 600;
  border-color: #DED7FF;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
  background: #E2DCFF; color: {C['purple_deep']};
}}
section[data-testid="stSidebar"] button p {{ font-size: 12.5px; }}
.mm-profile {{
  display: flex; align-items: center; gap: 9px; border-top: 1px solid {C['border']};
  margin-top: 16px; padding: 14px 6px 2px;
}}
.mm-avatar {{
  width: 30px; height: 30px; border-radius: 50%; background: {C['lavender']};
  color: {C['purple_deep']}; font-size: 10.5px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}}
.mm-profile-name {{ font-size: 11.5px; font-weight: 600; }}
.mm-profile-sub {{ font-size: 9.5px; color: {C['muted']}; }}

/* ----------------------------------------------------------------- header */
.mm-crumbs {{ font-size: 10.5px; color: #9A9DA6; margin-bottom: 7px; letter-spacing: 0.01em; }}
.mm-crumbs b {{ color: {C['purple_deep']}; font-weight: 600; }}
.mm-title {{ font-size: 26px; line-height: 1.2; font-weight: 700; letter-spacing: -0.025em; }}
.mm-sub {{ font-size: 12.5px; color: {C['muted']}; margin-top: 4px; margin-bottom: 18px; }}

/* ------------------------------------------------------------------ cards */
/* Streamlit 1.50: a bordered container is a vertical block inside a layout wrapper. */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"] {{
  background: {C['card']}; border: 1px solid {C['border']}; border-radius: 16px;
  box-shadow: 0 1px 2px rgba(23, 23, 23, 0.03); gap: 0.65rem;
}}
.mm-kpi {{
  background: {C['card']}; border: 1px solid {C['border']}; border-radius: 16px;
  padding: 15px 16px 13px; box-shadow: 0 1px 2px rgba(23,23,23,0.03); height: 100%;
}}
.mm-kpi-top {{ display: flex; align-items: center; justify-content: space-between; }}
.mm-kpi-label {{ font-size: 10.8px; color: {C['muted']}; font-weight: 500; }}
.mm-kpi-icon {{
  width: 27px; height: 27px; border-radius: 8px; display: flex;
  align-items: center; justify-content: center;
}}
.mm-kpi-value {{ font-size: 24px; font-weight: 700; letter-spacing: -0.03em; margin-top: 8px; line-height: 1.1; }}
.mm-kpi-help {{ font-size: 9.8px; color: #9A9DA6; margin-top: 4px; line-height: 1.35; }}

.mm-section-title {{ font-size: 13px; font-weight: 650; letter-spacing: -0.01em; }}
.mm-section-sub {{ font-size: 10.5px; color: {C['muted']}; margin-top: 1px; }}
.mm-badge {{
  display: inline-flex; align-items: center; gap: 4px; padding: 2.5px 8px;
  border-radius: 999px; font-size: 9.5px; font-weight: 600; white-space: nowrap;
}}
.mm-badge.lavender {{ background: {C['lavender']}; color: {C['purple_deep']}; }}
.mm-badge.green {{ background: #E7F7F0; color: #17835C; }}
.mm-badge.orange {{ background: #FEF1E7; color: #C46722; }}
.mm-badge.blue {{ background: #EDF1FD; color: #4864CC; }}
.mm-badge.pink {{ background: #FCEBF2; color: #C14E7B; }}
.mm-badge.grey {{ background: #F2F3F6; color: #6B6F78; }}
.mm-empty {{
  border: 1px dashed #DEDFE7; border-radius: 13px; background: #FCFCFE;
  padding: 22px 18px; text-align: center; color: {C['muted']}; font-size: 11.5px;
  line-height: 1.55;
}}
.mm-empty strong {{ display: block; color: {C['text']}; font-size: 12.5px; margin-bottom: 3px; }}
.mm-note {{ font-size: 10.5px; color: {C['muted']}; line-height: 1.55; }}
.mm-stat {{ display: flex; justify-content: space-between; align-items: baseline;
  padding: 7px 0; border-bottom: 1px solid #F1F1F5; font-size: 11px; }}
.mm-stat:last-child {{ border-bottom: none; }}
.mm-stat span {{ color: {C['muted']}; }}
.mm-stat b {{ font-weight: 600; }}
.mm-activity {{ display: flex; align-items: center; gap: 9px; padding: 7px 0;
  border-bottom: 1px solid #F2F2F6; font-size: 11px; }}
.mm-activity:last-child {{ border-bottom: none; }}
.mm-activity-time {{ margin-left: auto; color: #A0A3AC; font-size: 9.8px; white-space: nowrap; }}
.mm-swatch {{ display: inline-block; width: 100%; height: 38px; border-radius: 9px; }}
.mm-mini-label {{ font-size: 10px; color: {C['muted']}; font-weight: 500; }}
.mm-mini-value {{ font-size: 19px; font-weight: 700; letter-spacing: -0.02em; margin-top: 2px; }}
.mm-mini-help {{ font-size: 9.5px; color: #A0A3AC; margin-top: 2px; }}
.mm-result-label {{ font-size: 10.5px; color: {C['muted']}; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }}
.mm-result-value {{ font-size: 38px; font-weight: 700; letter-spacing: -0.04em; line-height: 1.05; margin: 5px 0 2px; }}
.mm-divider {{ border-top: 1px solid {C['border']}; margin: 4px 0 2px; }}

/* --------------------------------------------------------------- controls */
.stButton > button {{
  border-radius: 9px; font-weight: 600; font-size: 12px; padding: 0.42rem 0.9rem;
  border: 1px solid {C['border']}; background: white; color: {C['text']};
  box-shadow: none; transition: all 120ms ease;
}}
.stButton > button:hover {{ border-color: #D8D2F8; background: {C['hover']}; color: {C['purple_deep']}; }}
.stButton > button[kind="primary"] {{ background: {C['purple_deep']}; border-color: {C['purple_deep']}; color: white; }}
.stButton > button[kind="primary"]:hover {{ background: #574AC7; border-color: #574AC7; color: white; }}
.stDownloadButton > button {{
  border-radius: 9px; border: 1px solid {C['border']}; background: white;
  color: {C['text']}; font-size: 12px; font-weight: 600;
}}
.stDownloadButton > button:hover {{ background: {C['hover']}; border-color: #D8D2F8; color: {C['purple_deep']}; }}

div[data-baseweb="input"], div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] {{
  border: 1px solid {C['border']} !important; border-radius: 9px !important;
  background: white !important; box-shadow: none !important;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="textarea"]:focus-within {{
  border-color: {C['purple']} !important;
  box-shadow: 0 0 0 3px {C['lavender']} !important;
}}
label p {{ font-size: 11px !important; color: #4B4F58 !important; font-weight: 500 !important; }}
div[data-testid="stWidgetLabel"] p {{ margin-bottom: 2px; }}
div[data-baseweb="radio"] label p {{ font-size: 11.5px !important; }}
div[data-testid="stCheckbox"] label p {{ font-size: 11.5px !important; }}

/* ------------------------------------------------------------------ tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.1rem; border-bottom: 1px solid {C['border']}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.35rem 0.1rem 0.5rem; color: {C['muted']}; font-size: 12px; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {C['purple_deep']}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {C['purple_deep']}; height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ background: transparent; }}

/* --------------------------------------------------------------- surfaces */
div[data-testid="stExpander"] {{
  border: 1px solid {C['border']}; border-radius: 12px; background: white;
  box-shadow: none; overflow: hidden;
}}
div[data-testid="stExpander"] summary {{ font-size: 11.5px; font-weight: 600; color: #3F4248; }}
div[data-testid="stDataFrame"] {{
  border: 1px solid {C['border']}; border-radius: 12px; overflow: hidden;
  background: white;
}}
div[data-testid="stAlert"] {{ border-radius: 12px; font-size: 11.5px; border: 1px solid transparent; }}
div[data-testid="stAlert"]:has(svg[title="Info"]) {{ background: #F6F4FF; }}
hr {{ border-color: {C['border']}; margin: 1.1rem 0; }}
.stCaption, [data-testid="stCaptionContainer"] {{ font-size: 10.5px; color: {C['muted']}; }}
.stProgress > div > div > div > div {{ background: {C['purple_deep']}; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle, crumb="CRM-Adv"):
    st.markdown(
        f'<div class="mm-crumbs">{crumb} / <b>{title}</b></div>'
        f'<div class="mm-title">{title}</div>'
        f'<div class="mm-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section(title, subtitle=None):
    html = f'<div class="mm-section-title">{title}</div>'
    if subtitle:
        html += f'<div class="mm-section-sub">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


def kpi(label, value, help_text, icon_name="chart", tone="lavender"):
    tones = {
        "lavender": (C["lavender"], C["purple_deep"]),
        "green": ("#E7F7F0", "#1B8F65"),
        "orange": ("#FEF1E7", "#C46722"),
        "blue": ("#EDF1FD", "#4864CC"),
        "pink": ("#FCEBF2", "#C14E7B"),
    }
    bg, fg = tones.get(tone, tones["lavender"])
    st.markdown(
        f'<div class="mm-kpi">'
        f'<div class="mm-kpi-top"><div class="mm-kpi-label">{label}</div>'
        f'<div class="mm-kpi-icon" style="background:{bg};color:{fg}">{icon(icon_name)}</div>'
        f'</div>'
        f'<div class="mm-kpi-value">{value}</div>'
        f'<div class="mm-kpi-help">{help_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def badge(text, tone="lavender"):
    return f'<span class="mm-badge {tone}">{text}</span>'


def empty(title, body):
    st.markdown(
        f'<div class="mm-empty"><strong>{title}</strong>{body}</div>',
        unsafe_allow_html=True,
    )


def note(text):
    st.markdown(f'<div class="mm-note">{text}</div>', unsafe_allow_html=True)


def mini(label, value, help_text=""):
    st.markdown(
        f'<div class="mm-mini-label">{label}</div>'
        f'<div class="mm-mini-value">{value}</div>'
        + (f'<div class="mm-mini-help">{help_text}</div>' if help_text else ""),
        unsafe_allow_html=True,
    )


def stat_rows(rows):
    html = "".join(
        f'<div class="mm-stat"><span>{k}</span><b>{v}</b></div>' for k, v in rows
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------- figures ---

def fig_ax(width=6.4, height=3.0, grid=True):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(C["workspace"])
    ax.set_facecolor(C["workspace"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["border"])
    ax.tick_params(colors=C["muted"], length=0, pad=4)
    if grid:
        ax.yaxis.grid(True, color="#F0F0F5", linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, ax


def show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def grouped_bars(categories, series, width=6.4, height=3.0, value_labels=False):
    fig, ax = fig_ax(width, height)
    n = len(series)
    x = np.arange(len(categories))
    bar_w = 0.78 / max(n, 1)
    for i, (name, values) in enumerate(series.items()):
        color = MODEL_COLORS.get(name, SERIES[i % len(SERIES)])
        bars = ax.bar(x + i * bar_w, values, bar_w, label=name, color=color,
                      edgecolor="none", zorder=3)
        if value_labels:
            for b in bars:
                ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2,
                            b.get_height()), textcoords="offset points",
                            xytext=(0, 2), ha="center", fontsize=7, color=C["muted"])
    ax.set_xticks(x + bar_w * (n - 1) / 2, categories)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=n)
    fig.tight_layout()
    return fig


def grouped_series(categories, series, colors=None, width=6.4, height=3.0,
                   ylim=(0, 1.02), fmt="{:.2f}"):
    fig, ax = fig_ax(width, height)
    n = len(series)
    x = np.arange(len(categories))
    bar_w = 0.78 / max(n, 1)
    for i, (name, values) in enumerate(series.items()):
        color = (colors or SERIES)[i % len(colors or SERIES)]
        bars = ax.bar(x + i * bar_w, values, bar_w, label=name, color=color,
                      edgecolor="none", zorder=3)
        for b in bars:
            ax.annotate(fmt.format(b.get_height()),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7, color=C["muted"])
    ax.set_xticks(x + bar_w * (n - 1) / 2, categories)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=n)
    fig.tight_layout()
    return fig


def h_bars(labels, values, color=None, width=6.4, height=2.8, fmt="{:,.2f}"):
    fig, ax = fig_ax(width, height)
    y = np.arange(len(labels))
    colors = color if isinstance(color, list) else [color or C["purple"]] * len(labels)
    ax.barh(y, values, color=colors, edgecolor="none", height=0.6, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="#F0F0F5", linewidth=0.8)
    ax.yaxis.grid(False)
    for i, v in enumerate(values):
        ax.annotate(fmt.format(v), (v, i), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7.5, color=C["muted"])
    fig.tight_layout()
    return fig


def lines(x, series, width=6.4, height=3.0, colors=None, hline=None, vline=None):
    fig, ax = fig_ax(width, height)
    for i, (name, values) in enumerate(series.items()):
        ax.plot(x, values, label=name, color=(colors or SERIES)[i % len(colors or SERIES)],
                linewidth=2, marker="o", markersize=3.5, zorder=3)
    if hline is not None:
        ax.axhline(hline[0], color=C["border"], linestyle="--", linewidth=1, label=hline[1])
    if vline is not None:
        ax.axvline(vline[0], color=C["border"], linestyle="--", linewidth=1)
        ax.annotate(vline[1], (vline[0], 0.02), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(5, 4), fontsize=7,
                    color=C["muted"])
    ax.set_xticks(list(x))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=len(series))
    fig.tight_layout()
    return fig


def donut(labels, values, colors=None, center_top="", center_bottom="",
          width=3.4, height=3.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(C["workspace"])
    colors = colors or SERIES
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                       wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 2})
    ax.text(0, 0.08, center_top, ha="center", va="center", fontsize=16,
            fontweight="bold", color=C["text"])
    ax.text(0, -0.17, center_bottom, ha="center", va="center", fontsize=8,
            color=C["muted"])
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.94, 0.5),
              frameon=False, fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def box_grid(series, flags=None, ncols=3, width=11.5, row_height=1.75):
    """Small-multiple box plots, one per feature, in original units.

    ``series`` maps a display label to a numeric array; ``flags`` optionally
    maps the same label to a flagged count for the axis note.
    """
    import math

    n = len(series)
    rows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(width, row_height * rows),
                             dpi=150, squeeze=False)
    fig.patch.set_facecolor(C["workspace"])
    flags = flags or {}
    for ax, (label, values) in zip(axes.flat, series.items()):
        ax.set_facecolor(C["workspace"])
        ax.boxplot(values, vert=False, widths=0.5, whis=1.5,
                   patch_artist=True,
                   boxprops={"facecolor": C["lavender"], "edgecolor": C["purple_deep"],
                             "linewidth": 1.1},
                   medianprops={"color": C["purple_deep"], "linewidth": 1.6},
                   whiskerprops={"color": C["muted"], "linewidth": 0.9},
                   capprops={"color": C["muted"], "linewidth": 0.9},
                   flierprops={"marker": "o", "markersize": 3,
                               "markerfacecolor": C["orange"],
                               "markeredgecolor": "none", "alpha": 0.7})
        ax.set_yticks([])
        ax.set_title(label, fontsize=9.5, loc="left", pad=5)
        note = f"{len(values):,} values"
        if label in flags:
            note += f" · {flags[label]:,} flagged"
        ax.set_xlabel(note, fontsize=7.5, color=C["muted"])
        ax.tick_params(colors=C["muted"], labelsize=7.5, length=0, pad=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(C["border"])
        ax.xaxis.grid(True, color="#F0F0F5", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    return fig


def hist_grid(series, fences=None, log=False, ylabel="Records", ncols=3,
              width=11.5, row_height=1.85):
    """Small-multiple histograms with the IQR fences marked."""
    import math

    fences = fences or {}
    n = len(series)
    rows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(width, row_height * rows),
                             dpi=150, squeeze=False)
    fig.patch.set_facecolor(C["workspace"])
    for ax, (label, values) in zip(axes.flat, series.items()):
        ax.set_facecolor(C["workspace"])
        plot_values, axis_note = values, label
        bounds = fences.get(label)
        if log and values.min() >= 0:
            plot_values = np.log1p(values)
            if bounds is not None:
                bounds = np.log1p(np.maximum(bounds, 0))
            axis_note = f"log(1 + {label})"
        ax.hist(plot_values, bins=30, color=C["purple"], edgecolor="white",
                linewidth=0.5, zorder=3)
        if bounds is not None:
            for bound in bounds:
                ax.axvline(bound, color=C["orange"], linestyle="--",
                           linewidth=1, zorder=4)
        ax.set_title(axis_note, fontsize=9.5, loc="left", pad=5)
        ax.set_ylabel(ylabel, fontsize=7.5, color=C["muted"])
        ax.tick_params(colors=C["muted"], labelsize=7.5, length=0, pad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["border"])
        ax.yaxis.grid(True, color="#F0F0F5", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    return fig


def confusion(cm, title, width=2.9, height=2.4):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(C["workspace"])
    cm = np.asarray(cm)
    ax.imshow(cm, cmap=CONFUSION_CMAP)
    top = cm.max() if cm.max() else 1
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=11,
                color="white" if value > top / 2 else C["text"])
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_yticks([0, 1], ["No", "Yes"])
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    ax.set_title(title, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig
