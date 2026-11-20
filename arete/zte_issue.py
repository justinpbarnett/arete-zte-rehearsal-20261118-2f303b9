from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import gh, pilot, tracker, util

if TYPE_CHECKING:
    from .zte_github import GitHub, Identity

LABEL = "agent:zte"
LABELS = frozenset({LABEL, "documentation"})
MAX_BODY_BYTES = 16 * 1024
CLOSING_QUERY = """
query ($owner: String!, $repo: String!, $issue: Int!, $pr: Int!) {
  repository(owner: $owner, name: $repo) {
    databaseId id nameWithOwner owner { ...ActorId }
    defaultBranchRef { name }
    issue(number: $issue) {
      id number state stateReason closedAt
      timelineItems(last: 1, itemTypes: [CLOSED_EVENT, REOPENED_EVENT]) {
        nodes {
          __typename
          ... on ClosedEvent {
            createdAt actor { ...ActorId }
            closer {
              __typename
              ... on PullRequest { id number mergeCommit { oid } }
              ... on Commit { oid }
            }
          }
        }
      }
    }
    pullRequest(number: $pr) {
      id number body merged mergedAt mergedBy { ...ActorId }
      baseRefName headRefOid mergeCommit { oid } isCrossRepository
      closingIssuesReferences(first: 2) { totalCount nodes { id number } }
    }
  }
}
fragment ActorId on Actor {
  ... on User { databaseId }
  ... on Bot { databaseId }
}
"""


@dataclass(frozen=True)
class Claim:
    number: int
    node: str
    revision: str


def require_number(value: int) -> None:
    pilot.require(type(value) is int and 0 < value < 2**31, "invalid GitHub issue or PR number")


def admit(identity: Identity, number: int, row: dict) -> Claim:
    require_number(number)
    pilot.require(row.get("number") == number and "pull_request" not in row
                  and row.get("repository_url") == f"https://api.github.com/repos/{identity.name}"
                  and type(row.get("node_id")) is str and bool(row["node_id"])
                  and (row.get("user") or {}).get("id") == identity.owner,
                  "GitHub did not return the owner-authored issue in the approved repository")
    pilot.require(row.get("state") == "open" and row.get("locked") is False and row.get("comments") == 0,
                  "the issue is closed, locked, or has discussion outside its pinned revision")
    raw_labels = row.get("labels")
    pilot.require(isinstance(raw_labels, list) and all(isinstance(label, dict)
                  and type(label.get("name")) is str for label in raw_labels), "GitHub did not return the issue labels")
    labels = sorted(label["name"] for label in raw_labels)
    pilot.require(LABEL in labels and set(labels) <= LABELS and len(set(labels)) == len(labels),
                  "the issue is not in the dedicated delivery queue")
    title, body, updated = (row.get(key) for key in ("title", "body", "updated_at"))
    pilot.require(type(title) is str and 0 < len(title.encode()) <= 512
                  and type(body) is str and 0 < len(body.encode()) <= MAX_BODY_BYTES
                  and type(updated) is str and len(updated) == 20 and updated.endswith("Z"),
                  "the issue text or revision is missing or exceeds the delivery bound")
    try:
        util.parse_iso(updated)
    except ValueError as error:
        raise pilot.Held("GitHub returned an invalid issue timestamp") from error
    pilot.require(not tracker.waits_on(body), "this lane does not admit issues with dependencies")
    fields = (identity.number, identity.node, identity.owner, number, row["node_id"], title, body, labels, updated)
    revision = hashlib.sha256(json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return Claim(number, row["node_id"], revision)


def current(client: GitHub, claim: Claim) -> None:
    require_number(claim.number)
    pilot.require(admit(client.identity, claim.number, client.read(f"/issues/{claim.number}")) == claim,
                  "the approved GitHub issue revision changed")


def closing_keyword(claim: Claim) -> str:
    require_number(claim.number)
    return f"Closes #{claim.number}"


def squash_title(pr_number: int) -> str:
    require_number(pr_number)
    return f"{pilot.TASK_TITLE} (#{pr_number})"


def pr_body(claim: Claim) -> gh.Outgoing:
    return gh.outgoing("Verified one-line README correction.", link=closing_keyword(claim), mark=True)


def binding(identity: Identity, claim: Claim, pr_number: int, pr_node: str, head: str, repo: dict) -> tuple[dict, dict]:
    require_number(claim.number)
    require_number(pr_number)
    pilot.require((repo.get("databaseId"), repo.get("id"), repo.get("nameWithOwner"))
                  == (identity.number, identity.node, identity.name)
                  and repo.get("owner") == {"databaseId": identity.owner}
                  and repo.get("defaultBranchRef") == {"name": "main"},
                  "GitHub did not return the approved repository and default branch")
    issue, pull = repo.get("issue") or {}, repo.get("pullRequest") or {}
    pilot.require((issue.get("number"), issue.get("id")) == (claim.number, claim.node), "the GitHub issue changed")
    pilot.require((pull.get("number"), pull.get("id"), pull.get("headRefOid")) == (pr_number, pr_node, head)
                  and pull.get("baseRefName") == "main" and pull.get("isCrossRepository") is False,
                  "GitHub did not return the approved PR, head and branch")
    pilot.require(pull.get("closingIssuesReferences") == {
        "totalCount": 1, "nodes": [{"id": claim.node, "number": claim.number}],
    }, "the PR does not close exactly the approved issue")
    return issue, pull


def require_link(identity: Identity, claim: Claim, pr_number: int, pr_node: str, head: str, repo: dict) -> None:
    issue, pull = binding(identity, claim, pr_number, pr_node, head, repo)
    pilot.require(issue.get("state") == "OPEN" and pull.get("merged") is False
                  and pull.get("body") == str(pr_body(claim)), "the pre-merge issue or exact PR body changed")


def require_completed(identity: Identity, claim: Claim, pr_number: int, pr_node: str, head: str, release: str,
                      repo: dict) -> dict:
    issue, pull = binding(identity, claim, pr_number, pr_node, head, repo)
    actor = {"databaseId": identity.bot}
    pilot.require(pull.get("merged") is True and pull.get("mergedBy") == actor and pilot.SHA.fullmatch(release)
                  and pull.get("mergeCommit") == {"oid": release} and pull.get("mergedAt")
                  and pull.get("body") == str(pr_body(claim)),
                  "GitHub did not report the approved squash and actor")
    closed, merged = issue.get("closedAt"), pull["mergedAt"]
    pilot.require((issue.get("state"), issue.get("stateReason")) == ("CLOSED", "COMPLETED") and closed
                  and util.parse_iso(closed) >= util.parse_iso(merged),
                  "GitHub did not report a completed issue after the squash")
    events = (issue.get("timelineItems") or {}).get("nodes")
    pilot.require(isinstance(events, list) and len(events) == 1, "GitHub did not report the latest close or reopen event")
    event = events[0]
    pull_closer = {"__typename": "PullRequest", "id": pr_node, "number": pr_number, "mergeCommit": {"oid": release}}
    commit_closer = {"__typename": "Commit", "oid": release}
    pilot.require(event.get("__typename") == "ClosedEvent" and event.get("actor") == actor
                  and event.get("closer") in (pull_closer, commit_closer)
                  and event.get("createdAt") and util.parse_iso(event["createdAt"]) >= util.parse_iso(closed),
                  "the latest issue close was not caused by the exact squash PR or its commit")
    return {"issue": claim.number, "issue_node": claim.node, "pr": pr_number, "pr_node": pr_node,
            "head": head, "release": release, "actor": identity.bot, "closed_at": closed,
            "closed_event_at": event["createdAt"]}
