#!/usr/bin/env python3
"""
OpenCode Hook: Require explicit user permission before executing any code.

This hook intercepts the `execute` tool (and shell commands) and prompts the user
for confirmation before allowing code execution. If the user rejects, the execution
is blocked and development pauses until the user manually tests/runs the code.
"""

import json
import sys
import os

def main():
    # Read the hook input from stdin
    input_data = json.load(sys.stdin)
    
    tool_name = input_data.get("tool", "")
    tool_args = input_data.get("arguments", {})
    
    # Tools that should require permission
    restricted_tools = ["execute", "shell", "bash", "cmd", "python", "node"]
    
    # Check if this is a restricted tool
    if tool_name in restricted_tools:
        command = tool_args.get("command", "") or tool_args.get("code", "") or str(tool_args)
        
        # Print to stderr so user sees it (stdout is used for hook response)
        print(f"\n⚠️  Code execution requested:", file=sys.stderr)
        print(f"   Tool: {tool_name}", file=sys.stderr)
        print(f"   Command/Code: {command[:500]}{'...' if len(command) > 500 else ''}", file=sys.stderr)
        print("", file=sys.stderr)
        
        # Prompt for permission
        try:
            response = input("   Allow execution? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        
        if response not in ("y", "yes"):
            # Block execution by returning an error response
            result = {
                "allowed": False,
                "error": "Execution blocked by user. Please run the code manually for testing."
            }
            print(json.dumps(result))
            return
    
    # Allow execution
    result = {"allowed": True}
    print(json.dumps(result))

if __name__ == "__main__":
    main()