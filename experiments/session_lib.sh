# Helpers shared by the GPU-session run scripts (source from the repo root; the caller sets LOG):
#   stamp                  date and time for log lines
#   run NAME DONE CMD...   skip when DONE (a finished step's marker file) is given and exists; otherwise run CMD with
#                          its output appended to $LOG/NAME.log, print the exit code and return it
#   finished DIR           print DIR/meta.json if that run record says "status": "done" (nothing otherwise)
#   require_tag TAG        stop unless HEAD is TAG and no tracked file is modified
stamp() { date "+%F %T"; }
run() {
    local name=$1 done=$2; shift 2
    if [ -n "$done" ] && [ -e "$done" ]; then echo "$(stamp) $name: already done"; return 0; fi
    echo "$(stamp) $name: start"
    "$@" >> "$LOG/$name.log" 2>&1
    local code=$?
    echo "$(stamp) $name: exit $code"
    return $code
}
finished() { [ -f "$1/meta.json" ] && grep -q '"status": "done"' "$1/meta.json" && echo "$1/meta.json"; }
require_tag() {
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "$1^{commit}" 2>/dev/null)" ] || { echo "HEAD is not $1"; exit 1; }
    [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree not clean"; exit 1; }
}
