from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.graph import run_text_claim_workflow
from app.sarvam import extract_claim_information


DEMO_TEXT = (
    "Meri car ka accident kal shaam ko Andheri mein hua tha. "
    "Bike wale ko hit hua aur bumper damage ho gaya. Kisi ko injury nahi hui."
)


def main() -> None:
    extraction = extract_claim_information(DEMO_TEXT)
    print("Sarvam-105B extraction")
    print(json.dumps(extraction.model_dump(), indent=2))

    result = run_text_claim_workflow(
        mobile_number="9876543210",
        customer_text=DEMO_TEXT,
        extractor=lambda _text: extraction,
    )
    print("\nClaimsVoice workflow result")
    print(
        json.dumps(
            {
                "customer_id": result.get("customer_id"),
                "policy_status": result.get("policy_status"),
                "coverage_status": result.get("coverage_status"),
                "claim_id": result.get("claim_id"),
                "claim_status": result.get("claim_status"),
                "next_action": result.get("next_action"),
                "response_message": result.get("response_message"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
