from src.streaming.stream_cuda_runtime import runtime_library_path


def test_paths_include_overlay_dependencies_and_preserve_existing(tmp_path):
    site=tmp_path/'site';lib=site/'nvidia/cudnn/lib';lib.mkdir(parents=True)
    assert runtime_library_path([str(site)],'/other')==str(lib)+':/other'


def test_path_configuration_is_idempotent(tmp_path):
    lib=tmp_path/'nvidia/cublas/lib';lib.mkdir(parents=True)
    first=runtime_library_path([str(tmp_path)],'/custom:/custom')
    assert runtime_library_path([str(tmp_path)],first)==first


def test_missing_optional_wheels_leave_existing_path_unchanged(tmp_path):
    assert runtime_library_path([str(tmp_path)],'/custom')=='/custom'
