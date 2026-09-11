"""Regression test for get_vision_explanation in app.py.

Run from the repo root with no test framework and no AWS credentials:

    python3 tests/test_bedrock.py

Bedrock is never called. The request is validated against the Converse input
shape botocore ships, and the response path runs through botocore's Stubber,
so a wrong key or a renamed field fails here rather than in production.
"""
import io, os, sys, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import boto3
from botocore.stub import Stubber, ANY
from botocore.exceptions import ClientError, ParamValidationError
from botocore.validate import ParamValidator
from PIL import Image
import translations as tr
warnings.filterwarnings("ignore")

SRC = open("app.py").read()
body = SRC[SRC.index("def get_vision_explanation("):SRC.index("def build_distribution(")]

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",
                       aws_access_key_id="x", aws_secret_access_key="y")
ns = {"io": io, "tr": tr, "bedrock_runtime": bedrock,
      "BEDROCK_MODEL_ID": "us.amazon.nova-lite-v1:0", "BEDROCK_MAX_TOKENS": 200}
exec(compile(body, "<app.py excerpt>", "exec"), ns)
explain = ns["get_vision_explanation"]

img = Image.new("RGB", (64, 48), (20, 30, 60))
fails = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ((" -- " + detail) if detail and not ok else ""))
    if not ok: fails.append(name)

def reply(text):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 400, "outputTokens": 60, "totalTokens": 460},
            "metrics": {"latencyMs": 700}}

print("1. request shape validates against the real Converse API model")
op = bedrock.meta.service_model.operation_model("Converse")

# (a) capture the exact params my code sends, then validate them against the
#     real input shape botocore ships for Converse.
sent = {}
def spy(**kw):
    sent.update(kw)
    return reply("captured")
ns["bedrock_runtime"] = type("C", (), {"converse": staticmethod(spy)})()
explain(img, "Electronic")
ns["bedrock_runtime"] = bedrock
try:
    ParamValidator().validate(sent, op.input_shape)
    check("params accepted by botocore's Converse input shape", True)
except ParamValidationError as e:
    check("params accepted by botocore's Converse input shape", False, str(e)[:200])

# (b) run the real client through Stubber, which validates my *response*
#     parsing against the real output shape.
with Stubber(bedrock) as sb:
    sb.add_response("converse", reply("A neon street at night. It fits electronic."))
    out = explain(img, "Electronic")
    sb.assert_no_pending_responses()

check("returns the model's text", out == "A neon street at night. It fits electronic.", repr(out))
check("modelId passed through", sent.get("modelId") == "us.amazon.nova-lite-v1:0")
check("maxTokens honoured", sent.get("inferenceConfig", {}).get("maxTokens") == 200)

content = sent["messages"][0]["content"]
check("image block sent as raw bytes, not base64",
      isinstance(content[0]["image"]["source"]["bytes"], (bytes, bytearray)),
      type(content[0]["image"]["source"]["bytes"]).__name__)
check("image is a real JPEG (magic bytes)", content[0]["image"]["source"]["bytes"][:2] == b"\xff\xd8")
check("declared format matches the payload", content[0]["image"]["format"] == "jpeg")
check("image precedes the text block", "image" in content[0] and "text" in content[1])

print("\n2. prompt construction")
def prompt_for(**kw):
    grabbed = {}
    def spy(**p):
        grabbed.update(p)
        return reply("ok")
    ns["bedrock_runtime"] = type("C", (), {"converse": staticmethod(spy)})()
    explain(img, kw.pop("genre", "Electronic"), **kw)
    ns["bedrock_runtime"] = bedrock
    return grabbed["messages"][0]["content"][1]["text"]

en = prompt_for()
ar = prompt_for(lang="ar", genre="إلكترونية")
gr = prompt_for(grad=True)
check("English prompt asks for English", tr.PROMPT_LANGUAGE["en"] in en)
check("Arabic prompt carries the Arabic instruction", tr.PROMPT_LANGUAGE["ar"] in ar)
check("Arabic prompt names the localized genre", "إلكترونية" in ar)
check("grad prompt references the heatmap", "Grad-CAM" in gr and "heatmap" in gr)
check("non-grad prompt does not", "Grad-CAM" not in en)
check("unknown language falls back to English", tr.PROMPT_LANGUAGE["en"] in prompt_for(lang="zz"))

print("\n3. failure degrades to None instead of raising")
for code, msg in [("AccessDeniedException", "model access not enabled"),
                  ("ThrottlingException", "rate exceeded"),
                  ("ValidationException", "bad model id")]:
    def boom(**kw):
        raise ClientError({"Error": {"Code": code, "Message": msg}}, "Converse")
    ns["bedrock_runtime"] = type("C", (), {"converse": staticmethod(boom)})()
    try:
        r = explain(img, "Electronic")
        check(f"{code} -> None", r is None, repr(r))
    except Exception as e:
        check(f"{code} -> None", False, f"raised {type(e).__name__}")
    ns["bedrock_runtime"] = bedrock

def malformed(**kw):
    return {"output": {"message": {"content": []}}}
ns["bedrock_runtime"] = type("C", (), {"converse": staticmethod(malformed)})()
try:
    check("empty content block -> None", explain(img, "Electronic") is None)
except Exception as e:
    check("empty content block -> None", False, f"raised {type(e).__name__}")
ns["bedrock_runtime"] = bedrock

print("\n4. no OpenAI left anywhere")
check("app.py has no openai import", "openai" not in SRC.lower())
check("no OPENAI_API_KEY fetch", "OPENAI_API_KEY" not in SRC)
check("requirements.txt drops openai",
      not any(l.strip().startswith("openai") for l in open("requirements.txt")))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
