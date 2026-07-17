import pytest
from PIL import Image

from game_asset_api.postprocess import (
    copy_selected_frames,
    frame_indices,
    wan_source_frame_count,
    write_sprite_sheet,
)


def test_wan_source_frame_count_rounds_up_to_a_four_frame_group():
    assert wan_source_frame_count(8) == 9


@pytest.mark.parametrize("frame_count", [1, 0, -1])
def test_wan_source_frame_count_rejects_frame_counts_below_two(frame_count):
    with pytest.raises(ValueError, match="frame_count must be at least 2"):
        wan_source_frame_count(frame_count)


def test_frame_indices_selects_evenly_distributed_source_frames():
    assert frame_indices(9, 8) == [0, 1, 2, 3, 5, 6, 7, 8]


@pytest.mark.parametrize(
    ("source_count", "target_count", "message"),
    [
        (1, 1, "target_count must be at least 2"),
        (0, 2, "source_count must be at least 1"),
        (2, 3, "source_count must be at least target_count"),
    ],
)
def test_frame_indices_rejects_invalid_counts(source_count, target_count, message):
    with pytest.raises(ValueError, match=message):
        frame_indices(source_count, target_count)


def test_write_sprite_sheet_composes_rgba_frames_in_row_major_order(tmp_path):
    frames = [
        Image.new("RGBA", (2, 2), (255, 0, 0, 0)),
        Image.new("RGBA", (2, 2), (0, 255, 0, 128)),
        Image.new("RGBA", (2, 2), (0, 0, 255, 255)),
    ]
    destination = tmp_path / "sheets" / "sprite.png"

    path, columns, rows = write_sprite_sheet(frames, destination)

    assert path == destination
    assert columns == 2
    assert rows == 2
    with Image.open(path) as sheet:
        assert sheet.mode == "RGBA"
        assert sheet.size == (4, 4)
        assert sheet.getpixel((2, 0)) == (0, 255, 0, 128)


def test_write_sprite_sheet_rejects_an_empty_frame_list(tmp_path):
    with pytest.raises(ValueError, match="at least one frame"):
        write_sprite_sheet([], tmp_path / "sprite.png")


def test_write_sprite_sheet_rejects_mismatched_frame_dimensions(tmp_path):
    frames = [Image.new("RGBA", (2, 2)), Image.new("RGBA", (3, 2))]

    with pytest.raises(ValueError, match="same dimensions"):
        write_sprite_sheet(frames, tmp_path / "sprite.png")


def test_copy_selected_frames_preserves_order_and_alpha(tmp_path):
    source_paths = []
    for index, color in enumerate(
        [(255, 0, 0, 0), (0, 255, 0, 128), (0, 0, 255, 255)]
    ):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGBA", (2, 2), color).save(path)
        source_paths.append(path)

    destination = tmp_path / "selected"
    copied_paths = copy_selected_frames(source_paths, [2, 0], destination)

    assert copied_paths == [destination / "000.png", destination / "001.png"]
    with Image.open(copied_paths[0]) as first:
        assert first.mode == "RGBA"
        assert first.getpixel((0, 0)) == (0, 0, 255, 255)
    with Image.open(copied_paths[1]) as second:
        assert second.mode == "RGBA"
        assert second.getpixel((0, 0)) == (255, 0, 0, 0)


def test_copy_selected_frames_reads_all_sources_before_overwriting_them(tmp_path):
    source_paths = [tmp_path / "000.png", tmp_path / "001.png"]
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(source_paths[0])
    Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(source_paths[1])

    copied_paths = copy_selected_frames(source_paths, [1, 0], tmp_path)

    assert copied_paths == source_paths
    with Image.open(source_paths[0]) as first:
        assert first.getpixel((0, 0)) == (0, 255, 0, 255)
    with Image.open(source_paths[1]) as second:
        assert second.getpixel((0, 0)) == (255, 0, 0, 255)


@pytest.mark.parametrize("selected_indices", [[-1], [3]])
def test_copy_selected_frames_rejects_invalid_source_indices(tmp_path, selected_indices):
    source = tmp_path / "source.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(source)

    with pytest.raises(ValueError, match="invalid source frame index"):
        copy_selected_frames([source], selected_indices, tmp_path / "selected")


def test_copy_selected_frames_requires_a_source_path_for_each_selection(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(source)

    with pytest.raises(ValueError, match="not enough source paths"):
        copy_selected_frames([source], [0, 0], tmp_path / "selected")
