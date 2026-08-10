"""Virtual clothing try-on page."""

from __future__ import annotations

import streamlit as st

from frontend.components.media import preview_image
from frontend.components.results import render_completed_job
from frontend.components.theme import render_hero
from frontend.config.settings import FrontendSettings, PROJECT_ROOT
from frontend.pages.common import (
    current_record,
    execute_generation,
    from_upload,
    reset_product,
    select_example,
)
from frontend.services.api_client import (
    BackendClient,
    GarmentUpload,
    UploadedImage,
)

MAX_GARMENTS = 8
GARMENT_TYPE_LABELS = {
    "T-shirt": "T-shirt / تی‌شرت",
    "Shirt": "Shirt / پیراهن",
    "Jacket": "Jacket / کت",
    "Coat": "Coat / پالتو",
    "Hoodie": "Hoodie / هودی",
    "Sweatshirt": "Sweatshirt / سویشرت",
    "Pants": "Pants / شلوار",
    "Jeans": "Jeans / شلوار جین",
    "Shorts": "Shorts / شلوارک",
    "Skirt": "Skirt / دامن",
    "Dress": "Dress / لباس",
    "Shoes": "Shoes / کفش",
    "Hat": "Hat / کلاه",
    "Watch": "Watch / ساعت",
    "Bag": "Bag / کیف",
    "Glasses": "Glasses / عینک",
    "Other": "Other / سایر",
}


def _init_garment_state() -> None:
    st.session_state.setdefault("clothing_garment_ids", [0])
    st.session_state.setdefault("clothing_next_garment_id", 1)


def _add_garment() -> None:
    garment_ids = list(st.session_state["clothing_garment_ids"])
    if len(garment_ids) >= MAX_GARMENTS:
        return
    next_id = int(st.session_state["clothing_next_garment_id"])
    st.session_state["clothing_garment_ids"] = [*garment_ids, next_id]
    st.session_state["clothing_next_garment_id"] = next_id + 1


def _remove_garment(garment_id: int) -> None:
    garment_ids = list(st.session_state["clothing_garment_ids"])
    if len(garment_ids) > 1:
        st.session_state["clothing_garment_ids"] = [
            item for item in garment_ids if item != garment_id
        ]


def _reset_garments() -> None:
    next_id = int(st.session_state["clothing_next_garment_id"])
    st.session_state["clothing_garment_ids"] = [next_id]
    st.session_state["clothing_next_garment_id"] = next_id + 1


def _selected_source(person_upload: object | None) -> UploadedImage | None:
    return from_upload(person_upload) or st.session_state.get(
        "clothing_example_source"
    )


def render(client: BackendClient, settings: FrontendSettings) -> None:
    _init_garment_state()
    render_hero(
        "Virtual clothing try-on",
        "Build a complete look from multiple references.",
        (
            "Upload one image per garment or accessory, then specify exactly "
            "which item should be transferred from each image."
        ),
    )
    token = st.session_state.get("clothing_reset_token", 0)
    controls, workspace = st.columns([0.86, 1.45], gap="large")

    with controls:
        with st.container(border=True):
            st.markdown("### Create a look")
            st.caption("PNG, JPG, or WebP · Up to 8 garment references")
            person_upload = st.file_uploader(
                "Person image",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"clothing_person_{token}",
            )
            source = _selected_source(person_upload)

            st.markdown("#### Garments & accessories")
            st.caption(
                "For every image, choose the single item that should be tried on."
            )
            garments: list[GarmentUpload] = []
            garment_ids = list(st.session_state["clothing_garment_ids"])
            example_reference = st.session_state.get(
                "clothing_example_reference"
            )

            for position, garment_id in enumerate(garment_ids):
                with st.container(border=True):
                    st.markdown(f"**Item {position + 1}**")
                    upload = st.file_uploader(
                        f"Garment image {position + 1}",
                        type=["png", "jpg", "jpeg", "webp"],
                        key=f"clothing_garment_{token}_{garment_id}",
                    )
                    garment_type = st.selectbox(
                        "What should be transferred from this image?",
                        options=list(GARMENT_TYPE_LABELS),
                        format_func=GARMENT_TYPE_LABELS.__getitem__,
                        key=f"clothing_type_{token}_{garment_id}",
                    )
                    if garment_type == "Other":
                        garment_type = st.text_input(
                            "Item name",
                            placeholder="e.g. necklace / گردنبند",
                            key=f"clothing_custom_type_{token}_{garment_id}",
                        ).strip()
                    image = from_upload(upload)
                    if image is None and position == 0:
                        image = example_reference
                    if image is not None:
                        garments.append(
                            GarmentUpload(
                                image=image,
                                garment_type=garment_type,
                            )
                        )
                    if len(garment_ids) > 1 and st.button(
                        "Remove this item",
                        key=f"remove_garment_{token}_{garment_id}",
                        width="stretch",
                    ):
                        _remove_garment(garment_id)
                        st.rerun()

            st.button(
                "+ Add another garment",
                key=f"add_garment_{token}",
                on_click=_add_garment,
                width="stretch",
                disabled=len(garment_ids) >= MAX_GARMENTS,
            )
            candidates = st.slider(
                "Candidate count",
                min_value=1,
                max_value=4,
                value=1,
                help=(
                    "Intermediate items use one candidate; this count applies "
                    "to the final item."
                ),
            )
            generate = st.button(
                "Generate try-on",
                type="primary",
                width="stretch",
                disabled=st.session_state.get(
                    "generation_in_progress",
                    False,
                ),
            )
            if st.button("Reset inputs", width="stretch"):
                _reset_garments()
                reset_product("clothing")

    if generate:
        if source is None:
            st.error("Please upload a person image.")
        elif not garments:
            st.error("Please upload at least one garment or accessory image.")
        elif any(not garment.garment_type for garment in garments):
            st.error("Please enter a name for every item marked as Other.")
        else:
            execute_generation(
                product="clothing",
                source=source,
                reference=garments[0].image,
                garments=garments,
                options={
                    "candidates_per_color": candidates,
                    "max_retries": 0,
                    "preserve_face": True,
                    "preserve_pose": True,
                    "preserve_background": True,
                },
                client=client,
                settings=settings,
            )

    with workspace:
        with st.container(border=True):
            st.markdown("### Preview & result")
            preview_image(source, "Person preview")
            if garments:
                st.markdown("#### Selected items")
                columns = st.columns(min(3, len(garments)))
                for index, garment in enumerate(garments):
                    with columns[index % len(columns)]:
                        st.image(garment.image.content, width="stretch")
                        st.caption(
                            f"{index + 1}. {GARMENT_TYPE_LABELS.get(garment.garment_type, garment.garment_type)}"
                        )
            else:
                st.info("Add garment references to preview the complete look.")

            record = current_record("clothing")
            if record:
                st.divider()
                render_completed_job(record)
            else:
                st.markdown("Your generated look will appear here.")

    _examples()


def _examples() -> None:
    with st.expander("Example images", expanded=False):
        st.caption("Load a ready-to-present person and garment pair.")
        person = PROJECT_ROOT / "inputs" / "persons" / "3.jpg"
        garment = (
            PROJECT_ROOT
            / "inputs"
            / "garments"
            / "gray_sweatshirt.png"
        )
        if person.is_file() and garment.is_file():
            left, right = st.columns(2)
            left.image(str(person), caption="Person", width="stretch")
            right.image(str(garment), caption="Garment", width="stretch")
            if st.button("Use clothing example", width="stretch"):
                select_example(
                    product="clothing",
                    source_path=person,
                    reference_path=garment,
                )
        else:
            st.info("Add example images to the backend inputs directory.")
