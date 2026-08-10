"""Image preview, zoom, and comparison components."""

from __future__ import annotations

import base64
import html
import uuid

import streamlit as st
import streamlit.components.v1 as components

from frontend.services.api_client import UploadedImage


def preview_image(image: UploadedImage | None, title: str) -> None:
    st.markdown(f'<div class="section-label">{html.escape(title)}</div>', unsafe_allow_html=True)
    if image is None:
        st.markdown(
            '<div class="empty-preview">Drop an image here to preview it.</div>',
            unsafe_allow_html=True,
        )
        return
    st.image(image.content, width="stretch")


def zoomable_image(image: bytes, mime_type: str, *, height: int = 540) -> None:
    """Render a click-to-zoom image without sending it to a third party."""

    encoded = base64.b64encode(image).decode("ascii")
    element_id = f"zoom-{uuid.uuid4().hex}"
    components.html(
        f"""
        <style>
          html,body{{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif}}
          .frame{{height:{height - 8}px;border-radius:18px;overflow:hidden;background:#0b101c;
            cursor:zoom-in;display:grid;place-items:center;position:relative}}
          .frame img{{width:100%;height:100%;object-fit:contain;transition:transform .25s ease}}
          .frame.zoomed{{overflow:auto;cursor:zoom-out;display:block}}
          .frame.zoomed img{{width:auto;height:auto;min-width:100%;max-width:none;transform:scale(1.35);
            transform-origin:top left}}
          .hint{{position:absolute;right:12px;bottom:12px;background:rgba(0,0,0,.62);color:white;
            padding:7px 10px;border-radius:10px;font-size:12px;pointer-events:none}}
        </style>
        <div id="{element_id}" class="frame" onclick="this.classList.toggle('zoomed')">
          <img src="data:{html.escape(mime_type)};base64,{encoded}" alt="Generated result" />
          <span class="hint">Click to zoom</span>
        </div>
        """,
        height=height,
        scrolling=False,
    )


def comparison_slider(
    before: UploadedImage,
    after: bytes,
    after_mime: str,
    *,
    before_label: str = "Before",
    after_label: str = "After",
    height: int = 520,
) -> None:
    """Render an interactive before/after reveal slider."""

    before_data = base64.b64encode(before.content).decode("ascii")
    after_data = base64.b64encode(after).decode("ascii")
    element_id = f"compare-{uuid.uuid4().hex}"
    components.html(
        f"""
        <style>
          html,body{{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif}}
          .compare{{height:{height - 10}px;position:relative;overflow:hidden;border-radius:18px;background:#0b101c}}
          .compare img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}}
          .after{{clip-path:inset(0 0 0 50%)}}
          .line{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:white;box-shadow:0 0 14px #000}}
          .knob{{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:42px;height:42px;
            border-radius:50%;background:white;color:#20263a;display:grid;place-items:center;font-weight:800;
            box-shadow:0 8px 24px rgba(0,0,0,.35)}}
          input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
          .tag{{position:absolute;top:14px;padding:7px 10px;border-radius:9px;background:rgba(0,0,0,.62);
            color:white;font-size:12px;font-weight:700}}
          .left{{left:14px}} .right{{right:14px}}
        </style>
        <div id="{element_id}" class="compare">
          <img src="data:{html.escape(before.content_type)};base64,{before_data}" alt="Before" />
          <img class="after" src="data:{html.escape(after_mime)};base64,{after_data}" alt="After" />
          <div class="line"></div><div class="knob">↔</div>
          <span class="tag left">{html.escape(before_label)}</span>
          <span class="tag right">{html.escape(after_label)}</span>
          <input aria-label="Before and after comparison" type="range" min="0" max="100" value="50"
            oninput="const p=this.value+'%';const r=this.parentElement;r.querySelector('.after').style.clipPath='inset(0 0 0 '+p+')';r.querySelector('.line').style.left=p;r.querySelector('.knob').style.left=p;" />
        </div>
        """,
        height=height,
        scrolling=False,
    )
