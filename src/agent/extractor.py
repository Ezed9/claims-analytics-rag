import re
import sys
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import extract_with_tool


class IncidentType(str, Enum):
    SCREEN_DAMAGE = "SCREEN_DAMAGE"
    LIQUID_DAMAGE = "LIQUID_DAMAGE"
    BATTERY = "BATTERY"
    LOST_STOLEN = "LOST_STOLEN"
    HARDWARE_MALFUNCTION = "HARDWARE_MALFUNCTION"
    OTHER_DAMAGE = "OTHER_DAMAGE"


class ExtractedClaimInfo(BaseModel):
    device_brand: str | None = Field(default=None)
    device_model: str | None = Field(default=None)
    incident_type: IncidentType
    damage_symptoms: list[str] = Field(default_factory=list)
    incident_date: str | None = Field(default=None)
    carrier: str | None = Field(default=None)
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str = Field(default="rule_based_fallback")


TOOL_SCHEMA = {
    "name": "extract_claim_entities",
    "description": "Extract structured claim entities from a free-text warranty claim narrative.",
    "input_schema": ExtractedClaimInfo.model_json_schema(),
}

DEVICE_PATTERNS = [
    (r"iphone\s*16\s*pro\s*max", "Apple", "iPhone 16 Pro Max"),
    (r"iphone\s*16\s*pro", "Apple", "iPhone 16 Pro"),
    (r"iphone\s*16\s*plus", "Apple", "iPhone 16 Plus"),
    (r"iphone\s*16", "Apple", "iPhone 16"),
    (r"iphone\s*15\s*pro\s*max", "Apple", "iPhone 15 Pro Max"),
    (r"iphone\s*15\s*pro", "Apple", "iPhone 15 Pro"),
    (r"iphone\s*15\s*plus", "Apple", "iPhone 15 Plus"),
    (r"iphone\s*15", "Apple", "iPhone 15"),
    (r"iphone\s*14\s*pro\s*max", "Apple", "iPhone 14 Pro Max"),
    (r"iphone\s*14\s*pro", "Apple", "iPhone 14 Pro"),
    (r"iphone\s*14\s*plus", "Apple", "iPhone 14 Plus"),
    (r"iphone\s*14", "Apple", "iPhone 14"),
    (r"iphone\s*13\s*pro\s*max", "Apple", "iPhone 13 Pro Max"),
    (r"iphone\s*13\s*pro", "Apple", "iPhone 13 Pro"),
    (r"iphone\s*13\s*mini", "Apple", "iPhone 13 mini"),
    (r"iphone\s*13", "Apple", "iPhone 13"),
    (r"galaxy\s*s24", "Samsung", "Galaxy S24"),
    (r"galaxy\s*s23", "Samsung", "Galaxy S23"),
    (r"pixel\s*9", "Google", "Pixel 9"),
    (r"pixel\s*8", "Google", "Pixel 8"),
]

INCIDENT_KEYWORDS = [
    (IncidentType.LOST_STOLEN, ["stolen", "lost", "missing", "theft", "break-in"]),
    (IncidentType.LIQUID_DAMAGE, ["liquid", "water", "spill", "submerged", "rain", "wet"]),
    (IncidentType.BATTERY, ["battery", "drain", "shut down", "swollen", "swelling"]),
    (IncidentType.SCREEN_DAMAGE, ["screen", "spiderweb", "touch input", "touch response"]),
    (
        IncidentType.HARDWARE_MALFUNCTION,
        ["camera", "speaker", "sensor", "face id", "fingerprint", "microphone"],
    ),
]

CARRIER_KEYWORDS = ["Verizon", "AT&T", "T-Mobile", "Apple"]


def _match_device(claim_text: str) -> tuple[str | None, str | None]:
    lowered = claim_text.lower()
    for pattern, brand, model in DEVICE_PATTERNS:
        if re.search(pattern, lowered):
            return brand, model
    return None, None


def _match_incident_type(claim_text: str) -> IncidentType:
    lowered = claim_text.lower()
    for incident_type, keywords in INCIDENT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return incident_type
    return IncidentType.OTHER_DAMAGE


def _match_symptoms(claim_text: str, incident_type: IncidentType) -> list[str]:
    lowered = claim_text.lower()
    symptoms = []
    for _, keywords in INCIDENT_KEYWORDS:
        for keyword in keywords:
            if keyword in lowered:
                symptoms.append(keyword)
    return symptoms or [incident_type.value.lower().replace("_", " ")]


def _match_carrier(claim_text: str) -> str | None:
    for carrier in CARRIER_KEYWORDS:
        if carrier.lower() in claim_text.lower():
            return carrier
    return None


def extract_rule_based(claim_text: str, carrier_hint: str | None = None) -> ExtractedClaimInfo:
    device_brand, device_model = _match_device(claim_text)
    incident_type = _match_incident_type(claim_text)
    symptoms = _match_symptoms(claim_text, incident_type)
    carrier = carrier_hint or _match_carrier(claim_text)
    confidence = 0.55 if device_model else 0.4
    return ExtractedClaimInfo(
        device_brand=device_brand,
        device_model=device_model,
        incident_type=incident_type,
        damage_symptoms=symptoms,
        carrier=carrier,
        confidence=confidence,
        extraction_method="rule_based_fallback",
    )


def extract_claim_info(
    claim_text: str, carrier_hint: str | None = None, policy_handbook_text: str = ""
) -> ExtractedClaimInfo:
    from src.config import has_active_llm

    if not has_active_llm():
        return extract_rule_based(claim_text, carrier_hint)

    system_blocks = [
        {
            "type": "text",
            "text": policy_handbook_text or "No policy handbook context available.",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                "You extract structured entities from a warranty claim narrative. "
                "Only extract information present in the text; never decide claim eligibility."
            ),
        },
    ]
    try:
        result = extract_with_tool(system_blocks, claim_text, TOOL_SCHEMA)
        extracted = dict(result["extracted"])
        extracted.setdefault("carrier", carrier_hint)
        extracted["extraction_method"] = result["llm_provider"]
        return ExtractedClaimInfo.model_validate(extracted)
    except Exception:
        return extract_rule_based(claim_text, carrier_hint)


if __name__ == "__main__":
    info = extract_claim_info("Customer's iPhone 15 has a cracked screen on Verizon.")
    print(info.model_dump_json(indent=2))
