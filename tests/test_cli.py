from cli import expand_color_arguments


def test_space_separated_colors_are_expanded() -> None:
    assert expand_color_arguments(
        ["run", "--colors", "red", "blue", "#000000", "--max-retries", "1"]
    ) == [
        "run",
        "--colors",
        "red",
        "--colors",
        "blue",
        "--colors",
        "#000000",
        "--max-retries",
        "1",
    ]
