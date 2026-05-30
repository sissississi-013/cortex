"""Cortex — Brain Tools for Agents + Autonomous Neuroscience Researcher.

Entry points:
    # Run the MCP server (for any agent to connect)
    python -m cortex.main server
    python -m cortex.main server --http 8742

    # Run the autonomous research agent
    python -m cortex.main research "How does the brain process speech vs music?"

    # Run a quick demo
    python -m cortex.main demo
"""

from __future__ import annotations

import argparse
import asyncio
import sys


DEMO_QUESTIONS = [
    "How does the human brain differentially process speech vs. music, and which cortical regions are uniquely activated by each?",
    "What brain regions are involved in fear processing, and how does this relate to anxiety disorders?",
    "How does the default mode network contribute to episodic memory encoding?",
]


def cmd_server(args):
    """Start the Cortex MCP server."""
    from cortex.mcp_server import mcp

    if args.http:
        port = args.http
        print(f"[Cortex MCP] Starting HTTP server on port {port}")
        print(f"[Cortex MCP] Connect any MCP client to http://localhost:{port}")
        mcp.run(transport="http", host="0.0.0.0", port=port)
    else:
        mcp.run()


def cmd_research(args):
    """Run the autonomous research agent."""
    from cortex.signals import init_raindrop

    init_raindrop()

    from cortex.agent import run_cortex

    result = asyncio.run(run_cortex(
        research_question=args.question,
        model=args.model,
        verbose=True,
    ))

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"\nReport saved to {args.output}")


def cmd_demo(args):
    """Run a demo research session."""
    question = DEMO_QUESTIONS[0]

    from cortex.signals import init_raindrop
    init_raindrop()

    from cortex.agent import run_cortex

    print("\n" + "=" * 70)
    print("CORTEX DEMO — Autonomous Neuroscience Research")
    print("=" * 70)
    print(f"\nResearch Question:\n  {question}\n")

    result = asyncio.run(run_cortex(
        research_question=question,
        model=args.model,
        verbose=True,
    ))


def cmd_tools_test(args):
    """Quick test of brain tools (no LLM needed)."""
    import json

    from cortex.tools.tribe_fmri import predict_fmri_activation
    from cortex.tools.nimare_literature import search_literature, meta_analyze, decode_brain_region
    from cortex.tools.mne_eeg import analyze_eeg

    print("=" * 60)
    print("CORTEX BRAIN TOOLS — Quick Test")
    print("=" * 60)

    print("\n1. predict_fmri('a person speaking English')")
    r = predict_fmri_activation("a person speaking English")
    print(f"   Profile: {r['matched_profile']}")
    print(f"   Peak: {r['peak_region']['label']} ({r['peak_region']['activation_intensity']})")
    print(f"   Laterality: {r['laterality']}")
    print(f"   Top 3: {', '.join(x['label'] for x in r['top_activated_regions'][:3])}")

    print("\n2. search_literature('speech perception')")
    r = search_literature("speech perception")
    print(f"   Studies: {r['n_studies_matching']}")
    print(f"   Top cluster: {r['peak_activation_clusters'][0]['label']}")

    print("\n3. meta_analyze('music perception')")
    r = meta_analyze("music perception")
    print(f"   Method: {r['method']}")
    print(f"   Significant clusters: {len(r['significant_clusters'])}")
    for c in r["significant_clusters"][:3]:
        print(f"     - {c['anatomical_label']} (z={c['peak_z_score']}, p={c['p_value']:.2e})")

    print("\n4. decode_brain_region(-22, -4, -18)")
    r = decode_brain_region(-22, -4, -18)
    print(f"   Region: {r['region_label']}")
    print(f"   Top function: {r['top_function']}")
    print(f"   Decoded: {', '.join(d['term'] for d in r['decoded_functions'][:3])}")

    print("\n5. analyze_eeg('erp', 'auditory')")
    r = analyze_eeg("erp", "auditory")
    print(f"   Components found: {len(r['components'])}")
    for c in r["components"]:
        print(f"     - {c['name']} at {c['latency_ms']}ms ({c['amplitude_uV']} µV)")

    print("\n" + "=" * 60)
    print("All tools working!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="Cortex: Brain Tools for Agents + Autonomous Neuroscience Researcher",
    )
    subparsers = parser.add_subparsers(dest="command")

    sp_server = subparsers.add_parser("server", help="Start the Cortex MCP server")
    sp_server.add_argument("--http", type=int, default=None, help="HTTP port (default: stdio)")

    sp_research = subparsers.add_parser("research", help="Run autonomous research")
    sp_research.add_argument("question", help="Research question to investigate")
    sp_research.add_argument("--model", default="gpt-4.1", help="LLM model to use")
    sp_research.add_argument("--output", "-o", help="Save report to file")

    sp_demo = subparsers.add_parser("demo", help="Run a demo research session")
    sp_demo.add_argument("--model", default="gpt-4.1", help="LLM model to use")

    sp_test = subparsers.add_parser("test-tools", help="Quick test of brain tools")

    args = parser.parse_args()

    if args.command == "server":
        cmd_server(args)
    elif args.command == "research":
        cmd_research(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "test-tools":
        cmd_tools_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
