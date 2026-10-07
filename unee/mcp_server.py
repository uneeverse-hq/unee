"""MCP server: lets AI agents (Claude, Cursor, any MCP client) call Unee's local decisions as tools.

    unee serve --model unee-0.8b-Q4_K_M.gguf          # in one terminal
    python -m unee.mcp_server                          # stdio MCP server; UNEE_URL defaults to http://127.0.0.1:8000

Claude Desktop / Claude Code config:
    {"mcpServers": {"unee": {"command": "python", "args": ["-m", "unee.mcp_server"]}}}

Typical uses: an agent asks Unee whether a proposed action is safe, whether a page tries to inject instructions,
or which tool or team a request belongs to, all locally, in milliseconds, at no API cost.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from unee.client import Client

server = MCPServer(name="unee", instructions=(
    "Unee is a small local decision model. Use `decide` to pick one option with probabilities, `yes_no` for a "
    "calibrated yes/no, and `rate` to place an input on an ordered scale. Inputs are treated as data."))
client = Client(os.environ.get("UNEE_URL", "http://127.0.0.1:8000"))


@server.tool(description="Pick the option that fits the input best; returns the choice, a probability per option "
                         "and, with explain=true, a one-sentence reason.")
def decide(input: str, question: str, options: dict[str, str], explain: bool = False) -> dict:
    return client.choice(input, question, options, explain=explain)


@server.tool(description="Probability (0-1) that the answer to a yes/no question about the input is yes.")
def yes_no(input: str, question: str) -> float:
    return client.noul(input, question)


@server.tool(description="Rate the input on an ordered scale (lowest level first); returns the expected level and "
                         "a probability per level.")
def rate(input: str, question: str, levels: list[str]) -> dict:
    return client.score(input, question, levels)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
