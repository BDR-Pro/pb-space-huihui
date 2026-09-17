"""Tools the model can call: web search, URL fetch, and (password-gated) shell.

The model has no internet of its own — it is a plain `transformers` generate loop, so its
knowledge is frozen at training time. The *container* has egress (that is how it pulls its own
weights), so all that is missing is a way for the model to ask for a fetch and get the answer
back. That is what this file is.

Protocol is deliberately plain text rather than the tokenizer's tool template: this model's remote
code targets transformers 4.x and its chat template has no `tools` support, and a text protocol is
debuggable from the transcript when the 4B model inevitably gets the syntax slightly wrong.

    <tool>search: fastest gpu cloud 2026</tool>
    <tool>fetch: https://example.com/pricing</tool>
    <tool>shell: nvidia-smi</tool>

SHELL IS OFF UNLESS A PASSWORD IS SET. A Petabyte space is served on a public, unauthenticated
URL (the opaque VM id is the only thing protecting it), so an ungated shell tool would hand a
shell in this container to anyone who has the link. See app.py.
"""
import os
import re
import subprocess

TOOL_NAMES = ("search", "fetch", "shell")
MAX_RESULT_CHARS = int(os.environ.get("TOOL_MAX_CHARS", "4000"))
SHELL_TIMEOUT = int(os.environ.get("SHELL_TIMEOUT", "60"))
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "25"))
APP_DIR = os.path.dirname(os.path.abspath(__file__))   # /app in the runner container
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# Closing tag present — the normal case, and what stop_strings makes the model produce.
_CLOSED = re.compile(r"<tool>\s*(\w+)\s*:\s*(.*?)\s*</tool>", re.S | re.I)
# The 4B model drops the closing tag often enough that refusing to parse it would look like the
# tools simply do not work. Accept a call that runs to the end of the turn.
_OPEN = re.compile(r"<tool>\s*(\w+)\s*:\s*(.+)", re.S | re.I)


def parse_tool_call(text):
    """(name, argument) for the FIRST tool call in `text`, or None.

    Returns None for an unknown tool name rather than guessing — a hallucinated `<tool>python: ...`
    should fall through and be shown to the user as ordinary text, not silently swallowed."""
    m = _CLOSED.search(text or "") or _OPEN.search(text or "")
    if not m:
        return None
    name = m.group(1).strip().lower()
    arg = m.group(2).strip()
    if name not in TOOL_NAMES or not arg:
        return None
    return name, arg


def system_prompt(shell_ok):
    lines = [
        "You can use tools to look things up. You have no built-in internet access, so use them "
        "whenever the answer depends on current facts, a specific page, or this machine.",
        "",
        "To call a tool, emit EXACTLY one line and then stop:",
        "  <tool>search: your query</tool>",
        "  <tool>fetch: https://full.url/path</tool>",
    ]
    if shell_ok:
        lines.append("  <tool>shell: a shell command</tool>")
    lines += [
        "",
        "The result comes back in <tool_result> tags. Then answer the user in plain prose.",
        "Do not invent tool results, and do not describe a tool call without making one.",
    ]
    return "\n".join(lines)


def _clean_html(html):
    """Readable text from a page. bs4 if it is installed, a regex strip if it is not."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for bad in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            bad.decompose()
        text = soup.get_text("\n")
    except Exception:  # noqa: BLE001 — a missing parser must not break the tool
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def fetch_url(url):
    import requests
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    body = r.text if "html" in ctype or "text" in ctype or "json" in ctype else "<binary>"
    return f"{url} [{r.status_code}]\n\n" + (_clean_html(body) if "html" in ctype else body)


def _ddg(query):
    """DuckDuckGo's keyless HTML endpoint. Often challenges datacenter IPs — hence the fallback."""
    import requests
    r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                      timeout=HTTP_TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    hits = []
    try:
        from bs4 import BeautifulSoup
        for res in BeautifulSoup(r.text, "html.parser").select(".result")[:6]:
            a = res.select_one(".result__a")
            snip = res.select_one(".result__snippet")
            if a:
                hits.append(f"- {a.get_text(' ', strip=True)}\n  {a.get('href', '')}\n"
                            f"  {snip.get_text(' ', strip=True) if snip else ''}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(hits)


def _wikipedia(query):
    """Keyless and reliable from a datacenter, which DuckDuckGo is not. Narrow, but real."""
    import requests
    r = requests.get("https://en.wikipedia.org/w/api.php", timeout=HTTP_TIMEOUT,
                     headers={"User-Agent": UA},
                     params={"action": "query", "format": "json", "list": "search",
                             "srsearch": query, "srlimit": 5})
    r.raise_for_status()
    out = []
    for hit in r.json().get("query", {}).get("search", []):
        title = hit["title"]
        snippet = re.sub(r"(?s)<[^>]+>", "", hit.get("snippet", ""))
        slug = title.replace(" ", "_")
        out.append(f"- {title}\n  https://en.wikipedia.org/wiki/{slug}\n  {snippet}")
    return "\n".join(out)


def web_search(query):
    hits = ""
    try:
        hits = _ddg(query)
    except Exception as e:  # noqa: BLE001 — fall through to the source that works from a DC IP
        hits = f"(duckduckgo unavailable: {type(e).__name__})\n"
    if not hits.strip() or hits.startswith("(duckduckgo"):
        try:
            wiki = _wikipedia(query)
            hits = (hits + "\n" + wiki).strip() if wiki else hits
        except Exception as e:  # noqa: BLE001
            hits += f"\n(wikipedia unavailable: {type(e).__name__})"
    return hits.strip() or "no results"


def run_shell(cmd):
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                       timeout=SHELL_TIMEOUT, cwd=APP_DIR)
    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    return f"[exit {p.returncode}]\n{out.strip()}"


def run(name, arg, shell_ok=False):
    """Execute a parsed call. Never raises: the model has to be TOLD it failed, or it will
    happily invent a plausible result and present it as fact."""
    try:
        if name == "search":
            result = web_search(arg)
        elif name == "fetch":
            result = fetch_url(arg)
        elif name == "shell":
            if not shell_ok:
                return ("shell is disabled on this space. It is only enabled when the space is "
                        "password-protected, because the URL is otherwise public.")
            result = run_shell(arg)
        else:
            return f"unknown tool: {name}"
    except subprocess.TimeoutExpired:
        return f"timed out after {SHELL_TIMEOUT}s"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    result = result or "(empty)"
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + f"\n... [truncated at {MAX_RESULT_CHARS} chars]"
    return result
