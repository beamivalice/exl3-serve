#!/bin/bash
# Static guard for .github/workflows/release.yml event gating.
#
# The release workflow triples as (1) the tag/dispatch RELEASE pipeline,
# (2) the dry-run packaging check, and (3) the PR packaging build that
# signs + notarizes the CLI tarball WITHOUT releasing. The class of bug this
# pins: someone edits a step's `if:` and a PR suddenly creates a tag or a
# GitHub release — or the opposite, PR builds silently stop notarizing and
# the artifact regresses to unsigned.
#
# Hermetic — parses the YAML, no network, no runners.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'EOF'
import re, sys, yaml

FAIL = 0
def check(cond, msg):
    global FAIL
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        FAIL = 1

wf = yaml.safe_load(open(".github/workflows/release.yml"))

# YAML 1.1 parses the bare key `on` as boolean True.
triggers = wf.get("on", wf.get(True, {}))
check("pull_request" in triggers, "pull_request trigger present")
check("push" in triggers and "workflow_dispatch" in triggers,
      "tag-push + workflow_dispatch triggers still present")

job = wf["jobs"]["build"]

# Fork PRs have no secrets — the job must skip itself, not fail at cert import.
job_if = str(job.get("if", ""))
check("github.event.pull_request.head.repo.full_name == github.repository" in job_if,
      "job-level fork-PR guard present")

steps = {s.get("name", ""): s for s in job["steps"]}

def step_if(name):
    check(name in steps, f"step exists: {name}")
    return str(steps.get(name, {}).get("if", ""))

# Release-only steps must be OFF for PRs.
rel_if = step_if("Create Release")
check("pull_request" in rel_if and "!=" in rel_if,
      "Create Release gated off for pull_request")
check("workflow_dispatch" in step_if("Create tag (manual dispatch)"),
      "tag creation restricted to workflow_dispatch")

# ── The pre_release checkbox exists so a build can be cut WITHOUT consuming the
# version number: it tags v<YY.M.N>-pre-release.<n>, which the `^vYY.M.[0-9]+$`
# match that picks N never sees, so the plain vYY.M.N is still there to cut
# later. Two halves, both load-bearing:
#   - the suffix must be applied where the tag is MINTED (the version step), not
#     at the release step, or the tag consumes the number anyway;
#   - the release must still be created as a DRAFT and must NOT set `prerelease`
#     itself — flipping that flag is a deliberate manual step.
inputs = triggers.get("workflow_dispatch", {}).get("inputs", {})
check("pre_release" in inputs, "workflow_dispatch offers a pre_release checkbox")
check(inputs.get("pre_release", {}).get("default") in (False, "false"),
      "pre_release defaults to OFF")
version_step = next((s for s in job["steps"] if s.get("id") == "version"), {})
check("inputs.pre_release" in str(version_step.get("run", "")),
      "the pre-release suffix is applied where the tag is minted")
check("-pre-release" in str(version_step.get("run", "")),
      "the pre-release tag suffix is spelled in the version step")
rel_with = steps.get("Create Release", {}).get("with", {})
check(rel_with.get("draft") in (True, "true"),
      "the release is still created as a draft")
check("prerelease" not in rel_with,
      "the workflow never sets the prerelease flag itself (manual, by design)")

# Notarization must RUN on PRs — its gate may exclude dry_run but never PRs.
check("pull_request" not in step_if("Notarize CLI"),
      "Notarize CLI not excluded on pull_request")

sushi_builds = [s for s in job["steps"]
                    if s.get("name") == "Build sushi (Zig)"]
check(len(sushi_builds) == 1, "exactly one sushi release-artifact build")
check(any("-Dgit-sha=${{ github.sha }}" in str(s.get("run", ""))
          for s in sushi_builds),
      "release sushi build passes -Dgit-sha=${{ github.sha }}")

# The NAX static guard must run in the RELEASE pipeline itself — ci.yml
# checking the same cache key doesn't cover a cache-miss rebuild on the
# release runner, and that stage is what actually ships in the tarball.
nax_steps = [s for s in job["steps"]
             if "test_mlx_staged_nax.sh" in str(s.get("run", ""))]
check(len(nax_steps) == 1, "NAX metallib static guard step present")
check(nax_steps and "if" not in nax_steps[0],
      "NAX guard unconditional (runs on every event incl. PRs)")

# The PR build's output must be uploaded as an artifact.
upload = [s for s in job["steps"]
          if s.get("uses", "").startswith("actions/upload-artifact")]
check(any("pull_request" in str(s.get("if", "")) for s in upload),
      "artifact upload covers pull_request")

# ── CalVer timezone: the release ran at 01:26 UTC on Aug 1 while it was still
# Jul 31 locally, so CI minted 26.8.1 against a CHANGELOG and perf artifacts
# that all said 26.7.12. Runners are UTC, so YY.M must come from a pinned zone
# or it disagrees with the tree for a few hours around every month boundary.
wf_text = open(".github/workflows/release.yml").read()
check("date -u +%y" not in wf_text, "CalVer month is not computed in UTC")
check(re.search(r"TZ[=:]\s*[\"']?([A-Za-z_]+/[A-Za-z_]+)", wf_text) is not None,
      "release.yml pins the CalVer timezone")

# ── Third-party attribution must travel WITH the binary. The shipped binary
# links Apache-2.0 code (MTPLX/dflash/oMLX Metal kernels, jinja.cpp), and
# section 4 conditions redistribution on the recipient getting the license text
# and the NOTICE attributions. Nothing pinned that those files leave the repo,
# and for months they did not: the packaging path shipped the binary alone.
for f in ("LICENSE-APACHE-2.0", "NOTICE"):
    check(f in wf_text, f"release.yml packages {f} into the CLI tarball")

sys.exit(FAIL)
EOF
