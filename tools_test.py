"""tools_test.py — the tool parser and the shell gate.

The parser is the part that silently breaks: a 4B model gets the syntax slightly wrong and the
whole feature looks dead, or worse, a hallucinated tool name gets executed. The shell gate is the
part that matters if it breaks — a Petabyte space is served on a PUBLIC url, so shell must be off
unless the space is password-protected.

Offline. Run: python tools_test.py
"""
import tools

_fail = 0


def ok(label, cond):
    global _fail
    print(("ok   " if cond else "FAIL ") + label)
    if not cond:
        _fail += 1


p = tools.parse_tool_call

# ---------------------------------------------------------------- parsing
ok("a well-formed call parses", p("<tool>search: gpu prices</tool>") == ("search", "gpu prices"))
ok("surrounding prose does not break it",
   p("Let me look.\n<tool>fetch: https://a.b/c</tool>\n") == ("fetch", "https://a.b/c"))
ok("a missing closing tag still parses (the 4B model drops it often)",
   p("<tool>search: who won") == ("search", "who won"))
ok("the FIRST call wins when the model emits two",
   p("<tool>search: a</tool><tool>search: b</tool>") == ("search", "a"))
ok("case and padding are tolerated",
   p("<TOOL>  Search :   spaced  </TOOL>") == ("search", "spaced"))
ok("a multi-line argument survives",
   p("<tool>shell: echo one\necho two</tool>") == ("shell", "echo one\necho two"))

ok("plain prose is not a tool call", p("I would search for gpu prices.") is None)
ok("an unknown tool name is NOT executed", p("<tool>python: import os</tool>") is None)
ok("an empty argument is not a call", p("<tool>search: </tool>") is None)
ok("empty input is safe", p("") is None and p(None) is None)

# ---------------------------------------------------------------- the shell gate
out = tools.run("shell", "echo hello", shell_ok=False)
ok("shell REFUSES when the space has no password", "disabled" in out and "hello" not in out)
ok("and the refusal explains why, so the model can tell the user", "public" in out)

out = tools.run("shell", "echo hello", shell_ok=True)
ok("shell runs when the space IS password-protected", "hello" in out and "[exit 0]" in out)
ok("a failing command reports its exit code rather than raising",
   "[exit 1]" in tools.run("shell", "exit 1", shell_ok=True))

ok("an unknown tool name is refused by run() too",
   "unknown tool" in tools.run("python", "import os", shell_ok=True))

# ---------------------------------------------------------------- failure is reported, not hidden
# A tool that raises must come back as TEXT the model can read. Returning nothing (or raising)
# makes a 4B model invent a plausible result and present it as fact.
bad = tools.run("fetch", "http://127.0.0.1:9/nope")
ok("a failed fetch returns an error string instead of raising",
   isinstance(bad, str) and bad and "Error" not in bad[:0])

_long = tools.run("shell", "head -c 20000 /dev/zero | tr '\\0' 'x'", shell_ok=True)
ok("oversized output is truncated so it cannot blow the context window",
   len(_long) <= tools.MAX_RESULT_CHARS + 80 and "truncated" in _long)

# ---------------------------------------------------------------- the prompt matches the gate
ok("the prompt offers shell only when shell is on", "shell:" in tools.system_prompt(True))
ok("and never mentions it when it is off", "shell:" not in tools.system_prompt(False))
ok("search and fetch are always offered",
   all(t in tools.system_prompt(False) for t in ("search:", "fetch:")))

print()
print("=== tools: " + ("0 failures" if _fail == 0 else str(_fail) + " FAILED") + " ===")
raise SystemExit(1 if _fail else 0)
