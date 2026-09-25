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
import os, re, sys, yaml

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
# version: it tags v<MAJOR.MINOR.PATCH>-pre-release.<n>, so the plain version is
# still there to cut later. Two halves, both load-bearing:
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

# Without the Apple secrets the release ships ad-hoc signed: every Developer ID step is gated on the
# secrets being present, and packaging falls back to an ad-hoc signature instead of failing.
check("steps.signing.outputs.enabled" in step_if("Import signing certificate"),
      "certificate import runs only when the Apple secrets exist")
check("steps.signing.outputs.enabled" in step_if("Notarize CLI"),
      "notarization runs only when the Apple secrets exist")
check("SIGNING_IDENTITY=-" in str(steps.get("Package CLI binary", {}).get("run", "")),
      "packaging ad-hoc signs when there is no Developer ID")

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

# A public release must not ship a tracked file naming a development box's directories.
path_steps = [s for s in job["steps"]
              if "test_no_local_paths.sh" in str(s.get("run", ""))]
check(len(path_steps) == 1, "local-path guard step present")
check(path_steps and "if" not in path_steps[0],
      "local-path guard unconditional (runs on every event incl. PRs)")

# The PR build's output must be uploaded as an artifact.
upload = [s for s in job["steps"]
          if s.get("uses", "").startswith("actions/upload-artifact")]
check(any("pull_request" in str(s.get("if", "")) for s in upload),
      "artifact upload covers pull_request")

# ── SemVer. build.zig.zon's `.version` is the one version source; the version
# step sources release.sh so a dispatch and a tag push apply the same checks
# release.sh does, and no version comes from the clock.
wf_text = open(".github/workflows/release.yml").read()
version_run = str(version_step.get("run", ""))
check("release.sh" in version_run and "build.zig.zon" in version_run,
      "the version step reads build.zig.zon through release.sh")
check("release_version" in version_run and "tag_version_ok" in version_run,
      "the version step checks the CHANGELOG heading and a pushed tag")
check("date +%y" not in wf_text and "date -u +%y" not in wf_text,
      "no version is computed from the date")

import subprocess, tempfile
def sh(script, *args):
    r = subprocess.run(["bash", "-c", 'source ./release.sh; ' + script, "sh", *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip()

zon = re.search(r'\.version\s*=\s*"([^"]*)"', open("build.zig.zon").read())
check(zon is not None and sh('is_semver "$1"', zon.group(1))[0] == 0,
      "build.zig.zon's version is MAJOR.MINOR.PATCH")

with tempfile.TemporaryDirectory() as d:
    def release_version(heading, zon_version):
        cl, zf = os.path.join(d, "CHANGELOG.md"), os.path.join(d, "build.zig.zon")
        open(cl, "w").write(f"# Changelog\n\n{heading}\n\n- a bullet\n\n## v0.9.0 — Older\n")
        open(zf, "w").write(f'.{{\n    .name = .sushi,\n    .version = "{zon_version}",\n}}\n')
        return sh('release_version "$1" "$2"', cl, zf)
    check(release_version("## v1.0.0 — First", "1.0.0") == (0, "1.0.0"),
          "a v1.0.0 heading matching build.zig.zon is the release version")
    check(release_version("## v26.9.1 — CalVer", "1.0.0")[0] != 0,
          "a 26.9.x CHANGELOG heading is refused")
    check(release_version("## v26.9.1 — CalVer", "26.9")[0] != 0,
          "a two-part version is refused")
    check(release_version("## Unreleased", "1.0.0")[0] != 0,
          "an Unreleased top entry is refused even with an older version below it")
    check(release_version("## v1.1.0 — Next", "1.0.0")[0] != 0,
          "a heading that disagrees with build.zig.zon is refused")
for tag, ok in (("1.0.0", True), ("1.0.0-pre-release.2", True), ("26.9.1", False),
                ("1.0.1", False), ("1.0.0-pre-release.0", False), ("1x0x0", False)):
    check((sh('tag_version_ok "$1" 1.0.0', tag)[0] == 0) == ok,
          f"a pushed tag v{tag} is {'accepted' if ok else 'refused'} for 1.0.0")

# ── Third-party attribution must travel WITH the binary. The shipped binary
# links Apache-2.0 code (MTPLX/dflash/oMLX Metal kernels, jinja.cpp), and
# section 4 conditions redistribution on the recipient getting the license text
# and the NOTICE attributions. Nothing pinned that those files leave the repo,
# and for months they did not: the packaging path shipped the binary alone.
for f in ("LICENSE-APACHE-2.0", "NOTICE"):
    check(f in wf_text, f"release.yml packages {f} into the CLI tarball")
# ── A host that pins a release as its guest engine checks the tarball's sha256
# and reads guest.json from inside it; both must ship with every release.
package_run = str(steps.get("Package CLI binary", {}).get("run", ""))
check("--guest-manifest" in package_run and '"$STAGING/guest.json"' in package_run,
      "the packaged binary writes guest.json into the tarball")
check("steps.version.outputs.version" in package_run and "guest.json" in package_run.split("--guest-manifest", 1)[-1],
      "guest.json's version is checked against the release version")
tarball = "sushi-bin-macos-arm64.tar.gz"
sha_steps = [s for s in job["steps"]
             if "shasum -a 256" in str(s.get("run", "")) and f"{tarball}.sha256" in str(s.get("run", ""))]
check(len(sha_steps) == 1 and "if" not in sha_steps[0], "every build writes the tarball's .sha256")
check(f"{tarball}.sha256" in str(rel_with.get("files", "")), "the release publishes the .sha256")
check(any(f"{tarball}.sha256" in str(s.get("with", {}).get("path", "")) for s in upload),
      "PR / dry-run artifacts carry the .sha256")

# The packaging step copies them, so a tree without them cannot cut a release.
for f in ("LICENSE", "LICENSE-APACHE-2.0", "NOTICE"):
    check(os.path.isfile(f), f"{f} exists at the repo root")

# An attribution must point at a file a reader of this tree can open.
def expand_braces(p):
    m = re.search(r"\{([^}]*)\}", p)
    if not m:
        return [p]
    return [q for alt in m.group(1).split(",")
            for q in expand_braces(p[:m.start()] + alt + p[m.end():])]
notice_paths = set(re.findall(r"\b(?:src|lib|tests|scripts)/[A-Za-z0-9_./{},-]*[A-Za-z0-9_}/]",
                              open("NOTICE").read()))
for p in sorted(notice_paths):
    for q in expand_braces(p):
        check(os.path.exists(q), f"NOTICE names an existing path: {q}")

sys.exit(FAIL)
EOF
