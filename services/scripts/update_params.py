#!/usr/bin/env python3
"""
Param-file generator for Cohere.

Fetches live models and writes one compact param file per model
(``specs/cohere/<model>.json`` = ``{parameters}``) that the ``specs`` pipeline
re-renders ephemerally through ``templates/``. service_ids are preserved via
``<model>.service.json`` sidecars.

Usage: python scripts/update_params.py
"""

import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

# Provider Configuration
PROVIDER_NAME = "cohere"
PROVIDER_DISPLAY_NAME = "Cohere"
# Bare Cohere host.  The upstream ``base_url`` is the unversioned host;
# per-listing ``version_prefix`` parameters on the OpenAI-shape presets
# select either ``/v1`` (default — Cohere's native v1) or
# ``/compatibility/v1`` (Cohere's OpenAI-compatibility layer; see
# https://docs.cohere.com/v2/docs/compatibility-api).  The Cohere SDK
# preset uses bare ``/v2`` paths via the SDK's own URL handling.
API_BASE_URL = "https://api.cohere.ai"
ENV_API_KEY_NAME = "COHERE_API_KEY"

SCRIPT_DIR = Path(__file__).parent


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            r = httpx.get(
                "https://api.cohere.com/v2/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("models", [])
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("name", "")
            print(f"[{i}/{len(models)}] {model_id}")

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

    # Cohere advertises ``embed-*-image`` aliases (e.g.
    # ``embed-english-light-v3.0-image``) in its /v1/models listing but
    # the dated model behind them rejects both ``/embeddings`` text
    # input ("does not support text embeddings") and ``/v2/embed`` image
    # input ("does not support image embeddings").  Per Cohere's own
    # docs the canonical ``embed-*-v3.0`` and ``embed-v4.0`` models
    # accept both text and image input via ``input_type``, so the
    # ``-image`` aliases are redundant *and* broken — drop them.
    # Re-enable per-model if Cohere ever fixes this.
    _BROKEN_MODELS = frozenset({
        "embed-english-v3.0-image",
        "embed-english-light-v3.0-image",
        "embed-multilingual-v3.0-image",
        "embed-multilingual-light-v3.0-image",
    })

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict | None:
        """Build template variables for a model."""
        if model_id in self._BROKEN_MODELS:
            print(f"  Skipped: {model_id} (deprecated alias; Cohere returns 400 on both /embeddings and /v2/embed)")
            return None

        service_type = self._determine_service_type(model_id)
        model_lower = model_id.lower()
        # Audio-transcription detection.  The platform schema enum doesn't
        # have a transcription service_type today, so transcribe models
        # ride the ``llm`` slot and the listing template dispatches on
        # this flag.  Same pattern as Nebius's ``is_vision``.
        is_transcription = "transcribe" in model_lower or "whisper" in model_lower
        # Image-embedding detection.  ``embed-*-image`` models accept
        # image inputs and must hit Cohere's native ``/v2/embed``
        # endpoint with ``input_type=image`` — the OpenAI-compat
        # ``/embeddings`` rejects them ("does not support text
        # embeddings").
        is_image_embedding = (
            service_type == "embedding" and "image" in model_lower
        )
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Build details from LiteLLM data and model info
        details = {}
        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  Claude models are closed-source so parameter_count
        # is permanently null per the canonical helper.  metadata_sources
        # records provenance so reviewers can triage stale-value reports.
        canonical = ModelDataLookup.get_canonical_metadata(
            model_id,
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

        # BYOK: the customer's own key pays the provider directly, so the service
        # is free through the gateway.
        #
        # This plain description is what payout_price keeps (seller-facing). The
        # customer-facing listing cell is composed in listing.json.j2 from
        # pricing_note, into the "<amount> ~<intent> <PILL> | <note>" grammar
        # (unitysvc/unitysvc#1886); do not build it here, since this dict feeds
        # payout_price too.
        pricing = {"type": "constant", "price": "0", "description": "Free (BYOK)"}
        pricing_note = None
        if model_data and "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
            input_price = round(float(model_data["input_cost_per_token"]) * 1_000_000, 4)
            output_price = round(float(model_data["output_cost_per_token"]) * 1_000_000, 4)
            if "cache_read_input_token_cost" in model_data:
                cached_price = round(float(model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} / "
                    f"${self._format_price(cached_price)} "
                    f"per 1M input/output/cached tokens"
                )
            else:
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )
        # Closing pricing paragraph. The offering template renders
        # {{ description }} verbatim, so it is built here (not in the template).
        #
        # The price cell now carries "why free" itself — a BYOK pill beside the
        # amount, see listing.json.j2 — so this paragraph no longer repeats it
        # and states only the rate. It is NOT redundant with the cell's hover
        # note: a `title` attribute is unreachable on touch, absent from the API
        # and MCP catalog surfaces (`market_list_services` returns the
        # description, not a rendered tooltip), and not searchable, and for a
        # BYOK listing the upstream rate is the difference between free and real
        # money. Models the rate lookup does not cover keep the shorter
        # rate-less sentence rather than inventing a number.
        byok_para = (
            f"Pricing — usage is billed by {PROVIDER_DISPLAY_NAME} "
            f"directly{f' at {pricing_note}' if pricing_note else ''}."
        )

        capabilities = self._derive_capabilities(model_id, service_type, is_transcription)

        # Description suffix tracks the actual model nature so a
        # transcription model isn't described as "language model".
        if is_transcription:
            description_suffix = "audio transcription model"
        elif is_image_embedding:
            description_suffix = "image embedding model"
        elif service_type == "embedding":
            description_suffix = "embedding model"
        else:
            description_suffix = "language model"

        return {
            # Folder path under specs/ == listing.name == "<provider>/<model_id>"
            # (flat layout, #1263). populate_from_iterator preserves the slash.
            "name": f"{PROVIDER_NAME}/{model_id}",
            # Offering name is the bare upstream model_id
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} {description_suffix}.\n\n{byok_para}",
            "service_type": service_type,
            "capabilities": capabilities,
            "is_transcription": is_transcription,
            "is_image_embedding": is_image_embedding,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Bare upstream rate card (no "Service provider charges" prefix —
            # the copy that consumes it already names the biller), placed by
            # listing.json.j2 behind the `|` of the price cell.
            "pricing_note": pricing_note,
            # Listing fields
            "list_price": pricing,
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    def _determine_service_type(self, model_id: str) -> str:
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in ["embed", "embedding"]):
            return "embedding"
        if any(kw in model_lower for kw in ["rerank"]):
            return "embedding"  # rerank is a retrieval service, same access pattern
        return "llm"

    def _derive_capabilities(self, model_id: str, service_type: str,
                             is_transcription: bool) -> list[str]:
        """The platform capability this offering provides.

        One entry, from the platform vocabulary (unitysvc
        ``docs/capabilities.yml``): a capability names what the caller GETS
        from a call, so a model provides exactly one.

        Deliberately NOT included, and why:

        ``vision`` / ``tools``   Attributes.  They change what may appear in
                                 a chat request; they do not change what
                                 comes back.  A vision model and a plain
                                 chat model both answer with generated text.
        ``text-generation``      A HuggingFace pipeline tag, not a UnitySVC
                                 capability, and redundant with ``chat``.
        ``llm``                  An echo of ``service_type``.

        ``embed`` covers both text and image embedding — modality is
        ``input_formats``, and the outcome (a vector) is identical.  The
        ``/v2/embed`` vs ``/compatibility/v1/embeddings`` split is carried
        by ``is_image_embedding`` for example dispatch.
        """
        if is_transcription:
            return ["speech-transcribe"]
        if service_type == "embedding":
            # `_determine_service_type` files rerankers under the embedding
            # service_type (same access pattern); the capability is what
            # separates them — reordered documents out, not a vector.
            return ["rerank"] if "rerank" in model_id.lower() else ["embed"]
        return ["chat"]

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    write_params_from_iterator(
        iterator=source.iter_models(),
        output_dir=SCRIPT_DIR.parent / "specs",
    )


if __name__ == "__main__":
    main()
