import json

from c4zero_tools.version import current_version_info, repo_root


def test_version_manifest_has_required_keys():
    info = current_version_info()
    required = {
        "project_version",
        "cpp_abi_version",
        "python_tools_version",
        "dataset_schema_version",
        "checkpoint_schema_version",
        "model_config_version",
        "encoder_version",
        "game_rules_version",
        "action_mapping_version",
        "symmetry_version",
    }
    assert required <= set(info)


def test_packaged_version_manifest_matches_repo_manifest():
    packaged = current_version_info()
    repo = json.loads((repo_root() / "version_manifest.json").read_text(encoding="utf-8"))
    assert packaged == repo
