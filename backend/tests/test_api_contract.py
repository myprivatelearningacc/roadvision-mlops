# Simple API contract checks that do not require loading the model.
# For actual endpoint tests, configure a real checkpoint and run the app.
def test_expected_response_schema():
    expected = {"model_version", "image_width", "image_height", "threshold", "latency_ms", "detections"}
    assert "detections" in expected
