from app.tools import ToolRegistry, build_default_registry


def test_default_medication_tool_is_transparent_when_database_missing():
    registry = build_default_registry(database_url=None)

    result = registry.call("get_medication_adherence", {"date": "yesterday"})

    assert result["status"] == "unavailable"
    assert "database" in result["message"].lower()


def test_tool_registry_rejects_unknown_tool():
    registry = ToolRegistry()

    try:
        registry.call("missing_tool", {})
    except KeyError as exc:
        assert "missing_tool" in str(exc)
    else:
        raise AssertionError("Expected KeyError")

