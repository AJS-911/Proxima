"""Example usage of the Dynamic Tool System.

This file demonstrates how the dynamic tool system works:
1. Tools self-register using @register_tool decorator
2. LLM receives tool descriptions in its preferred format
3. LLM reasons about which tool(s) to use
4. System executes tools and returns results
5. LLM continues with the results

NO keyword matching - the LLM decides based on understanding!
"""

from proxima.agent.dynamic_tools import (
    # Core components
    get_tool_registry,
    get_current_context,
    get_tool_orchestrator,
    get_llm_tool_integration,
    
    # Configuration
    LLMToolConfig,
    LLMProvider,
    
    # For parsing tool calls
    ToolCall,
)


def demonstrate_tool_discovery():
    """Show how tools are discovered dynamically."""
    print("=" * 60)
    print("DYNAMIC TOOL DISCOVERY")
    print("=" * 60)
    
    registry = get_tool_registry()
    
    print(f"\nTotal registered tools: {len(registry)}")
    print("\nRegistered tools:")
    
    for registered in registry:
        defn = registered.definition
        print(f"  - {defn.name}")
        print(f"    Category: {defn.category.value}")
        print(f"    Description: {defn.description[:60]}...")
        print()


def demonstrate_llm_integration():
    """Show how tools are presented to the LLM."""
    print("=" * 60)
    print("LLM INTEGRATION - Tool Schemas")
    print("=" * 60)
    
    # Configure for Ollama (which uses OpenAI format)
    config = LLMToolConfig(
        provider=LLMProvider.OLLAMA,
        max_tools_per_request=10,
    )
    
    integration = get_llm_tool_integration(config)
    
    # Get tools in OpenAI format (used by Ollama, Gemini, etc.)
    tools = integration.get_tools_for_llm()
    
    print(f"\nGenerated {len(tools)} tool definitions for LLM")
    print("\nExample tool schema (OpenAI format):")
    
    import json
    if tools:
        print(json.dumps(tools[0], indent=2)[:500])
        print("...")


def demonstrate_tool_execution():
    """Show how tools are executed dynamically."""
    print("\n" + "=" * 60)
    print("DYNAMIC TOOL EXECUTION")
    print("=" * 60)
    
    orchestrator = get_tool_orchestrator()
    context = get_current_context()
    
    # Execute a tool directly
    print("\nExecuting 'list_directory' tool:")
    result = orchestrator.execute_single(
        "list_directory",
        {"path": ".", "pattern": "*.py"},
        context,
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.data:
        print(f"Found {len(result.data)} items")


def demonstrate_search_capability():
    """Show semantic search for tools."""
    print("\n" + "=" * 60)
    print("SEMANTIC TOOL SEARCH (No Keyword Matching!)")
    print("=" * 60)
    
    registry = get_tool_registry()
    
    # Search for tools related to a natural language query
    queries = [
        "I want to see what files are in my project",
        "commit my changes to git",
        "run a command in terminal",
        "read the contents of a file",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = registry.search_tools(query, limit=3)
        
        print("  Top matches:")
        for r in results:
            print(f"    - {r.tool.definition.name} (score: {r.relevance_score:.1f})")
            print(f"      Reason: {r.match_reason}")


def demonstrate_system_prompt_generation():
    """Show how to generate tool descriptions for system prompts."""
    print("\n" + "=" * 60)
    print("SYSTEM PROMPT GENERATION")
    print("=" * 60)
    
    config = LLMToolConfig(provider=LLMProvider.OLLAMA)
    integration = get_llm_tool_integration(config)
    
    prompt_section = integration.get_system_prompt_section()
    print("\nGenerated system prompt section:")
    print("-" * 40)
    print(prompt_section)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("   PROXIMA DYNAMIC TOOL SYSTEM - PHASE 1 DEMONSTRATION")
    print("=" * 60)
    print("""
This demonstrates the dynamic tool system that enables ANY LLM
(Ollama, Gemini, GPT, Claude, etc.) to:

1. DISCOVER tools dynamically (no hardcoding)
2. UNDERSTAND tool capabilities through schemas
3. SELECT tools based on reasoning
4. EXECUTE tools with proper context
5. PROCESS results for continued reasoning

The key insight: NO keyword matching!
The LLM uses reasoning to understand user intent and select tools.
""")
    
    try:
        demonstrate_tool_discovery()
        demonstrate_llm_integration()
        demonstrate_tool_execution()
        demonstrate_search_capability()
        demonstrate_system_prompt_generation()
        
        print("\n" + "=" * 60)
        print("PHASE 1 IMPLEMENTATION COMPLETE!")
        print("=" * 60)
        print("""
The dynamic tool system is now ready for integration with:
- Ollama (local models like Gemini 2.5 Flash)
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3.5, Claude 4, etc.)
- Google AI (Gemini Pro, etc.)
- Any provider supporting function calling

Next steps (Phase 2+):
- Integrate with existing AI assistant
- Add more tool categories (Backend, Analysis, etc.)
- Implement agent reasoning loop
""")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
