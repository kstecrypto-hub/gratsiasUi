from __future__ import annotations

from app.models.enums import ParticipantRole
from app.services.yeastar.interpretation import interpret_call_legs, safe_operator_channel


OPERATORS = [
    {
        "id": "operator-a",
        "yeastar_extension_id": "101",
        "extension_number": "2001",
    },
    {
        "id": "operator-b",
        "yeastar_extension_id": "102",
        "extension_number": "2002",
    },
]


def test_queue_leg_is_not_misattributed_and_answering_operator_is_identified() -> None:
    detail = {
        "timeline": [
            {
                "cdr_id": "queue-ring",
                "call_from_number": "+302100000000",
                "call_to_ext_id": "queue-600",
                "call_to_number": "600",
                "status": "NO ANSWER",
                "call_duration": 8,
            },
            {
                "cdr_id": "agent-answer",
                "call_from_ext_id": "queue-600",
                "call_from_number": "600",
                "call_to_ext_id": "101",
                "call_to_number": "2001",
                "status": "ANSWERED",
                "call_duration": 42,
                "talk_duration": 34,
            },
        ]
    }

    legs, participants = interpret_call_legs(detail, OPERATORS)

    assert [leg["yeastar_leg_id"] for leg in legs] == ["queue-ring", "agent-answer"]
    assert legs[1]["answered_by_extension_id"] == "101"
    assert [(item.operator_id, item.leg_id, item.role) for item in participants] == [
        ("operator-a", "agent-answer", ParticipantRole.ANSWERING_OPERATOR)
    ]


def test_transfer_preserves_both_operator_legs_and_explicit_roles() -> None:
    detail = {
        "timeline": [
            {
                "cdr_id": "initial-answer",
                "call_from_number": "+302100000000",
                "call_to_ext_id": "101",
                "call_to_number": "2001",
                "status": "ANSWERED",
            },
            {
                "cdr_id": "transfer-answer",
                "call_from_ext_id": "101",
                "call_from_number": "2001",
                "call_to_ext_id": "102",
                "call_to_number": "2002",
                "status": "ANSWERED",
            },
        ]
    }

    _, participants = interpret_call_legs(detail, OPERATORS)

    assert {(item.operator_id, item.leg_id, item.role) for item in participants} == {
        ("operator-a", "initial-answer", ParticipantRole.ANSWERING_OPERATOR),
        ("operator-a", "transfer-answer", ParticipantRole.CALLER),
        ("operator-b", "transfer-answer", ParticipantRole.ANSWERING_OPERATOR),
    }


def test_multiple_answering_operators_are_kept_as_distinct_participants() -> None:
    detail = {
        "transactions": [
            {
                "id": "leg-a",
                "call_from_number": "+302100000001",
                "call_to_ext_id": "101",
                "call_to_number": "2001",
                "status": "ANSWERED",
            },
            {
                "id": "leg-b",
                "call_from_number": "+302100000002",
                "call_to_ext_id": "102",
                "call_to_number": "2002",
                "status": "ANSWERED",
            },
        ]
    }

    _, participants = interpret_call_legs(detail, OPERATORS)

    answering = [item for item in participants if item.role == ParticipantRole.ANSWERING_OPERATOR]
    assert {(item.operator_id, item.leg_id) for item in answering} == {
        ("operator-a", "leg-a"),
        ("operator-b", "leg-b"),
    }


def test_channel_selection_requires_verified_one_to_one_stereo_separation() -> None:
    detail = {
        "timeline": [
            {
                "cdr_id": "leg-a",
                "call_from_ext_id": "101",
                "call_from_number": "2001",
                "call_to_number": "+302100000000",
                "status": "ANSWERED",
            },
            {
                "cdr_id": "leg-b",
                "call_from_number": "+302100000001",
                "call_to_ext_id": "102",
                "call_to_number": "2002",
                "status": "ANSWERED",
            },
        ]
    }
    _, participants = interpret_call_legs(detail, OPERATORS)
    caller = next(item for item in participants if item.operator_id == "operator-a")
    callee = next(item for item in participants if item.operator_id == "operator-b")

    assert safe_operator_channel(
        caller,
        channel_count=2,
        stereo_separated=True,
        one_to_one=True,
    ) == 0
    assert safe_operator_channel(
        callee,
        channel_count=2,
        stereo_separated=True,
        one_to_one=True,
    ) == 1
    assert safe_operator_channel(
        callee,
        channel_count=2,
        stereo_separated=True,
        one_to_one=True,
        was_transferred=True,
    ) is None
    assert safe_operator_channel(
        callee,
        channel_count=2,
        stereo_separated=True,
        one_to_one=True,
        operators_on_same_side=2,
    ) is None
