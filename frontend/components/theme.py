"""Premium light and dark visual system for the demo."""

from __future__ import annotations

import streamlit as st


def apply_theme(theme: str) -> None:
    """Apply an app-scoped visual theme without changing backend state."""

    dark = theme == "Dark"
    colors = {
        "bg": "#090D18" if dark else "#F5F7FB",
        "surface": "#111827" if dark else "#FFFFFF",
        "surface_2": "#172033" if dark else "#F0F4FA",
        "text": "#F7F9FC" if dark else "#172033",
        "muted": "#98A5BA" if dark else "#617087",
        "border": "rgba(255,255,255,.10)" if dark else "#DFE5EF",
        "shadow": (
            "0 20px 60px rgba(0,0,0,.30)"
            if dark
            else "0 20px 60px rgba(35,52,79,.10)"
        ),
    }
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {colors['bg']};
            --surface: {colors['surface']};
            --surface-2: {colors['surface_2']};
            --text: {colors['text']};
            --muted: {colors['muted']};
            --border: {colors['border']};
            --shadow: {colors['shadow']};
            --violet: #7657FF;
            --cyan: #16C7D9;
            --success: #26C281;
        }}
        .stApp {{
            background:
              radial-gradient(circle at 88% 8%, rgba(118,87,255,.12), transparent 28rem),
              radial-gradient(circle at 5% 90%, rgba(22,199,217,.08), transparent 30rem),
              var(--app-bg);
            color: var(--text);
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background: color-mix(in srgb, var(--surface) 94%, transparent);
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{ color: var(--text); }}
        .block-container {{
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}
        h1, h2, h3, p, label {{ color: var(--text); }}
        .app-kicker {{
            color: var(--cyan);
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: .55rem;
        }}
        .app-hero {{
            padding: 2rem 2.1rem;
            border: 1px solid var(--border);
            border-radius: 24px;
            background: linear-gradient(135deg, var(--surface), var(--surface-2));
            box-shadow: var(--shadow);
            margin-bottom: 1.5rem;
            overflow: hidden;
            position: relative;
        }}
        .app-hero::after {{
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -70px;
            top: -100px;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(118,87,255,.35), rgba(22,199,217,.15));
            filter: blur(12px);
        }}
        .app-hero h1 {{
            font-size: clamp(2rem, 4vw, 3.65rem);
            line-height: 1.04;
            letter-spacing: -.045em;
            max-width: 850px;
            margin: 0 0 .7rem 0;
        }}
        .app-hero p {{
            color: var(--muted);
            font-size: 1.05rem;
            max-width: 760px;
            margin: 0;
        }}
        .product-card, .panel-card, [data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: var(--border) !important;
            background: color-mix(in srgb, var(--surface) 97%, transparent);
            border-radius: 20px !important;
        }}
        .product-card {{
            min-height: 215px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
            box-shadow: 0 8px 30px rgba(24,35,54,.06);
        }}
        .product-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(118,87,255,.55);
            box-shadow: var(--shadow);
        }}
        .product-icon {{
            display: grid;
            place-items: center;
            width: 52px;
            height: 52px;
            border-radius: 16px;
            font-size: 1.55rem;
            background: linear-gradient(135deg, rgba(118,87,255,.18), rgba(22,199,217,.18));
            margin-bottom: 1.1rem;
        }}
        .product-card h3 {{ margin: 0 0 .5rem; font-size: 1.25rem; }}
        .product-card p {{ color: var(--muted); margin: 0; line-height: 1.6; }}
        .section-label {{
            color: var(--muted);
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .12em;
            margin: .4rem 0 .8rem;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .38rem .72rem;
            border-radius: 999px;
            background: rgba(38,194,129,.12);
            color: #26C281;
            font-size: .78rem;
            font-weight: 750;
            border: 1px solid rgba(38,194,129,.22);
        }}
        .metric-row {{ display:flex; gap:.65rem; flex-wrap:wrap; margin:.8rem 0; }}
        .metric-chip {{
            padding:.55rem .75rem;
            border-radius:12px;
            border:1px solid var(--border);
            color:var(--muted);
            background:var(--surface-2);
            font-size:.82rem;
        }}
        .metric-chip strong {{ color:var(--text); margin-left:.3rem; }}
        .empty-preview {{
            display:grid;
            place-items:center;
            min-height:230px;
            text-align:center;
            color:var(--muted);
            border:1px dashed var(--border);
            border-radius:18px;
            background:var(--surface-2);
            padding:1.4rem;
        }}
        .logo-mark {{
            width:44px;height:44px;border-radius:14px;display:grid;place-items:center;
            color:white;font-weight:850;font-size:1.05rem;
            background:linear-gradient(135deg,var(--violet),var(--cyan));
            box-shadow:0 10px 24px rgba(118,87,255,.28);
        }}
        .sidebar-brand {{display:flex;align-items:center;gap:.75rem;margin:.35rem 0 1.4rem;}}
        .sidebar-brand strong {{display:block;font-size:1rem;}}
        .sidebar-brand span {{display:block;color:var(--muted);font-size:.75rem;}}
        div.stButton > button, div.stDownloadButton > button {{
            border-radius: 12px;
            min-height: 2.8rem;
            border: 1px solid var(--border);
            font-weight: 720;
            transition: transform .18s ease, box-shadow .18s ease;
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            transform: translateY(-1px);
            border-color: var(--violet);
            box-shadow: 0 9px 25px rgba(118,87,255,.18);
        }}
        div.stButton > button[kind="primary"] {{
            color:white;
            border:0;
            background:linear-gradient(100deg,var(--violet),#5B7FFF 55%,var(--cyan));
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background:var(--surface-2);
            border:1px dashed var(--border);
            border-radius:16px;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{border-color:var(--violet);}}
        [data-testid="stAlert"] {{border-radius:14px;}}
        hr {{border-color:var(--border) !important;}}
        .app-footer {{
            text-align:center;color:var(--muted);font-size:.78rem;
            padding-top:3rem;
        }}
        @media (max-width: 760px) {{
            .block-container {{ padding: 1rem .85rem 3rem; }}
            .app-hero {{ padding: 1.45rem; border-radius: 19px; }}
            .product-card {{ min-height: auto; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <section class="app-hero">
          <div class="app-kicker">{kicker}</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_product_card(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="product-card">
          <div class="product-icon">{icon}</div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
