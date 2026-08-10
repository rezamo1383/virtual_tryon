"""Product dashboard."""

from __future__ import annotations

import streamlit as st

from frontend.components.theme import render_hero, render_product_card


def render() -> None:
    render_hero(
        "AI visualization suite",
        "Bring product ideas to life.",
        "Create client-ready visualizations in minutes with secure, tenant-aware AI generation.",
    )
    st.markdown('<div class="section-label">Choose a product</div>', unsafe_allow_html=True)
    clothing, wallpaper = st.columns(2, gap="large")
    with clothing:
        render_product_card(
            "◈",
            "Virtual Clothing Try-On",
            "Upload a person and a garment to create a realistic, identity-preserving outfit visualization.",
        )
        if st.button(
            "Open Clothing Studio  →",
            key="open_clothing",
            type="primary",
            width="stretch",
        ):
            st.session_state["page"] = "clothing"
            st.rerun()
    with wallpaper:
        render_product_card(
            "▦",
            "Wallpaper Visualization",
            "Preview a wallpaper pattern on the real walls of a room while preserving the original scene.",
        )
        if st.button(
            "Open Wallpaper Studio  →",
            key="open_wallpaper",
            type="primary",
            width="stretch",
        ):
            st.session_state["page"] = "wallpaper"
            st.rerun()

    st.markdown("### Designed for confident decisions")
    features = st.columns(3, gap="medium")
    for column, icon, title, text in zip(
        features,
        ("◎", "⌁", "↓"),
        ("Scene-aware", "Secure routing", "Presentation-ready"),
        (
            "Local preprocessing protects the composition before generation.",
            "Tenant credentials choose the correct backend configuration.",
            "Compare, zoom, and download full-resolution results instantly.",
        ),
        strict=True,
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"### {icon}  {title}")
                st.caption(text)
