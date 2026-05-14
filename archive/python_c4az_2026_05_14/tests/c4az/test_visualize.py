from pathlib import Path

from c4az.visualize import generate_visual_docs


def test_visual_docs_generation_is_scoped_and_complete(tmp_path: Path) -> None:
    output_dir = tmp_path / "visual"

    generate_visual_docs(output_dir)

    expected = {
        "index.html",
        "system_flow.mmd",
        "module_dependencies.mmd",
        "class_uml.mmd",
        "puct_sequence.mmd",
        "resnet_architecture.mmd",
        "data_lifecycle.mmd",
        "model_summary.json",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    index = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "src/c4az" in index
    assert "connect4_zero" not in index


def test_visual_docs_include_core_algorithm_objects(tmp_path: Path) -> None:
    output_dir = tmp_path / "visual"

    generate_visual_docs(output_dir)

    uml = (output_dir / "class_uml.mmd").read_text(encoding="utf-8")
    model = (output_dir / "resnet_architecture.mmd").read_text(encoding="utf-8")
    puct = (output_dir / "puct_sequence.mmd").read_text(encoding="utf-8")
    assert "class Position" in uml
    assert "class PUCTMCTS" in uml
    assert "class AlphaZeroNet" in uml
    assert "small: 229,879 params" in model
    assert "argmax Q + c_puct" in puct
