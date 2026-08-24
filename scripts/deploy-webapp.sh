#!/usr/bin/env bash
#
# Build the browser playground and publish it to GitHub Pages.
#
# This is a manual step, run from a machine that has the models: exported
# .onnx files are not committed, so there is nothing for CI to deploy. The
# usual sequence is
#
#     uv run tinyfacts export gpt_small        # -> webapp/public/models/
#     ./scripts/deploy-webapp.sh
#
# The script builds webapp/dist and pushes it as one commit on a publishing
# branch (gh-pages by default), whose tree mirrors dist exactly. That branch
# holds build output only — never edit it by hand, and never merge it anywhere.
#
# One-time setup on GitHub: Settings -> Pages -> Build and deployment ->
# Source: "Deploy from a branch", branch: gh-pages, folder: / (root).
#
set -euo pipefail

BRANCH="gh-pages"
REMOTE="origin"
DRY_RUN=0
SKIP_BUILD=0
ALLOW_NO_MODELS=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBAPP="$ROOT/webapp"
DIST="$WEBAPP/dist"
MODELS="$WEBAPP/public/models"

usage() {
	cat <<'USAGE'
Usage: scripts/deploy-webapp.sh [options]

Options:
  --branch <name>     Publishing branch (default: gh-pages)
  --remote <name>     Remote to push to (default: origin)
  --dry-run           Build and report what would be published, but do not
                      touch any branch or push
  --skip-build        Publish the existing webapp/dist as-is
  --allow-no-models   Publish even though webapp/public/models holds no models
  -h, --help          Show this message
USAGE
}

die() {
	echo "error: $*" >&2
	exit 1
}

step() {
	echo
	echo "==> $*"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--branch) BRANCH="${2:-}"; [[ -n "$BRANCH" ]] || die "--branch needs a value"; shift 2 ;;
		--remote) REMOTE="${2:-}"; [[ -n "$REMOTE" ]] || die "--remote needs a value"; shift 2 ;;
		--dry-run) DRY_RUN=1; shift ;;
		--skip-build) SKIP_BUILD=1; shift ;;
		--allow-no-models) ALLOW_NO_MODELS=1; shift ;;
		-h|--help) usage; exit 0 ;;
		*) usage >&2; die "unknown option: $1" ;;
	esac
done

command -v npm >/dev/null || die "npm is not installed"
git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 || die "$ROOT is not a git repository"

# ── check there is something worth deploying ──────────────────────────────────

step "Checking $MODELS"

shopt -s nullglob
onnx_files=("$MODELS"/*.onnx)
shopt -u nullglob

if [[ ${#onnx_files[@]} -eq 0 ]]; then
	if [[ $ALLOW_NO_MODELS -eq 0 ]]; then
		die "no .onnx models found in webapp/public/models.
Export one first:
    uv run tinyfacts export <model>
or pass --allow-no-models to publish the app without any."
	fi
	echo "no models found — publishing anyway (--allow-no-models)"
else
	for model in "${onnx_files[@]}"; do
		echo "  $(basename "$model")"
	done
	[[ -f "$MODELS/tokenizer.json" ]] || die "tokenizer.json is missing from webapp/public/models.
Re-run the export to regenerate it:
    uv run tinyfacts export <model>"
fi

# ── build ────────────────────────────────────────────────────────────────────

if [[ $SKIP_BUILD -eq 1 ]]; then
	step "Skipping build (--skip-build)"
	[[ -f "$DIST/index.html" ]] || die "no build found at webapp/dist — drop --skip-build"
else
	if [[ ! -d "$WEBAPP/node_modules" ]]; then
		step "Installing dependencies"
		if [[ -f "$WEBAPP/package-lock.json" ]]; then
			(cd "$WEBAPP" && npm ci)
		else
			(cd "$WEBAPP" && npm install)
		fi
	fi

	step "Building"
	(cd "$WEBAPP" && npm run build)
fi

[[ -f "$DIST/index.html" ]] || die "build produced no webapp/dist/index.html"

# Keep GitHub Pages from running the output through Jekyll.
touch "$DIST/.nojekyll"

source_commit="$(git -C "$ROOT" rev-parse --short HEAD)"
if ! git -C "$ROOT" diff --quiet HEAD -- "$WEBAPP"; then
	source_commit="$source_commit+dirty"
fi

if [[ $DRY_RUN -eq 1 ]]; then
	step "Dry run — would publish webapp/dist to $REMOTE/$BRANCH (from $source_commit)"
	(cd "$DIST" && find . -type f | sort | sed 's|^\./|  |')
	echo
	echo "total: $(du -sh "$DIST" | cut -f1)"
	exit 0
fi

# ── publish ──────────────────────────────────────────────────────────────────

git -C "$ROOT" remote get-url "$REMOTE" >/dev/null 2>&1 || die "no such remote: $REMOTE"

tmpdir="$(mktemp -d)"
worktree="$tmpdir/$BRANCH"

# A local branch full of build output is only worth keeping if it was already
# there; one this script creates is cleaned up on every exit path. The lease on
# the next run is checked against the remote-tracking ref either way.
had_local_branch=0
if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
	had_local_branch=1
fi

cleanup() {
	git -C "$ROOT" worktree remove --force "$worktree" >/dev/null 2>&1 || true
	rm -rf "$tmpdir"
	if [[ $had_local_branch -eq 0 ]]; then
		git -C "$ROOT" branch -D "$BRANCH" >/dev/null 2>&1 || true
	fi
}
trap cleanup EXIT

step "Preparing the $BRANCH branch"

# Does the remote already have the branch? `ls-remote --exit-code` answers 2
# for "no such ref", which is the one non-zero status that is not a failure.
ls_remote_status=0
git -C "$ROOT" ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1 || ls_remote_status=$?
case "$ls_remote_status" in
	0) remote_has_branch=1 ;;
	2) remote_has_branch=0 ;;
	*) die "could not reach $REMOTE" ;;
esac

# Start from whatever already exists — the remote branch first, then a local
# one (a deploy that was never pushed) — so the push stays a fast-forward where
# possible. With neither, the branch starts from scratch with no history.
if [[ $remote_has_branch -eq 1 ]]; then
	git -C "$ROOT" fetch --quiet "$REMOTE" "$BRANCH" || die "could not fetch $REMOTE/$BRANCH"
	git -C "$ROOT" worktree add --quiet -B "$BRANCH" "$worktree" FETCH_HEAD
elif [[ $had_local_branch -eq 1 ]]; then
	echo "$REMOTE/$BRANCH does not exist yet — continuing from the local $BRANCH"
	git -C "$ROOT" worktree add --quiet "$worktree" "$BRANCH"
else
	echo "$REMOTE/$BRANCH does not exist yet — creating it"
	git -C "$ROOT" worktree add --quiet --detach "$worktree"
	git -C "$worktree" checkout --quiet --orphan "$BRANCH"
	git -C "$worktree" rm -rq --cached . 2>/dev/null || true
fi

# The branch mirrors dist exactly: clear it out, then copy the build in.
find "$worktree" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$DIST/." "$worktree/"

step "Committing"
git -C "$worktree" add --all
if git -C "$worktree" diff --cached --quiet; then
	echo "$BRANCH is already up to date with this build — nothing to push"
	exit 0
fi
git -C "$worktree" commit --quiet -m "Deploy webapp from $source_commit"

step "Pushing to $REMOTE/$BRANCH"
if [[ $remote_has_branch -eq 1 ]]; then
	# The lease is against the ref we just fetched: refuse to clobber a deploy
	# someone else pushed in the meantime.
	git -C "$worktree" push --force-with-lease "$REMOTE" "$BRANCH"
else
	git -C "$worktree" push "$REMOTE" "$BRANCH"
fi

echo
echo "Done. GitHub Pages will publish $REMOTE/$BRANCH shortly."
