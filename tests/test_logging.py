from experience_learning.logging import JsonlEventLogger


def test_event_log_can_resume_at_checkpoint_offset(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path)
    logger.write("kept", value=1)
    checkpoint_offset = logger.byte_offset()
    logger.write("discarded", value=2)
    logger.truncate_to(checkpoint_offset)
    logger.write("replayed", value=3)
    logger.close()

    contents = path.read_text(encoding="utf-8")
    assert '"event": "kept"' in contents
    assert '"event": "discarded"' not in contents
    assert '"event": "replayed"' in contents
