"""Frozen constants for the ABCD Product Defect pilot."""

from __future__ import annotations


SOURCE_REPOSITORY = "asappresearch/abcd"
SOURCE_COMMIT = "6b8700ce67c6b37b062dd7a60abc76d7ef832a97"
SOURCE_FILES = (
    "data/abcd_v1.1.json.gz",
    "data/guidelines.json",
    "data/ontology.json",
    "LICENSE",
)
EXPECTED_SOURCE_METADATA = {
    "data/abcd_v1.1.json.gz": {
        "size": 36_985_084,
        "git_blob_sha": "c6fecb3aed30309ca3456ed93a0a19c98cade1a4",
        "sha256": "2bdf53ac359543dcdc38d55bc6513e78df120363f8f44870716e909f4606de15",
    },
    "data/guidelines.json": {
        "size": 95_293,
        "git_blob_sha": "62609c12e160d76ebf2e1820232f5c821198e372",
        "sha256": "9264557941df24fe075138a632a2345172971573124a354ccb2ceee2a12e4c2a",
    },
    "data/ontology.json": {
        "size": 5_770,
        "git_blob_sha": "29195b13fd33762bfe5402a49f231f5925837bd0",
        "sha256": "2e1c1d763518ba084ada7f7bc8b54f0b489c81da3b170875cf9340891e06524c",
    },
    "LICENSE": {
        "size": 1_071,
        "git_blob_sha": "f2dd20ad8cc771fd8d40be81b28880e6a7b4e28b",
        "sha256": "3ab7e179a7f13027b7bc64293541f0e9beacca3701cade67fb8fce78c2d9317b",
    },
}

FLOW_ID = "product_defect"
FLOW_TITLE = "Product Defect"
PRODUCT_DEFECT_SUBFLOWS = (
    "refund_initiate",
    "refund_update",
    "refund_status",
    "return_stain",
    "return_color",
    "return_size",
)
SUBFLOW_TITLES = {
    "refund_initiate": "Initiate Refund",
    "refund_update": "Update Refund",
    "refund_status": "Refund Status",
    "return_stain": "Return Due to Stain",
    "return_color": "Return Due to Color",
    "return_size": "Return Due to Size",
}
BUTTON_ACTION_OVERRIDES = {
    "membership privileges": "membership",
    "notify internal team": "notify-team",
}

EXPOSURES = ("full", "partial", "minimal", "no_policy")
TRAINING_REPETITIONS = len(EXPOSURES)
DEFAULT_ORDER_SEED = 20260827
TASK_ID = "abcd_product_defect_ast"

EXPECTED_PRODUCT_DEFECT_INVENTORY = {
    "train": {
        "conversation_count": 863,
        "action_turn_count": 3607,
        "policy_covered_action_count": 3309,
        "policy_clean_conversation_count": 701,
    },
    "dev": {
        "conversation_count": 102,
        "action_turn_count": 431,
        "policy_covered_action_count": 392,
        "policy_clean_conversation_count": 80,
    },
    "test": {
        "conversation_count": 105,
        "action_turn_count": 447,
        "policy_covered_action_count": 408,
        "policy_clean_conversation_count": 83,
    },
}
