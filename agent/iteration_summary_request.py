"""Iteration summaries obey the same request-local policy as normal turns."""


def managed_summary_call(agent, api_request_id, request, callback, *, retry_count):
    from agent import relay_llm
    from hermes_cli.middleware import apply_llm_request_middleware

    # Summary builders can start from cached agent.tools. Apply per-turn
    # restrictions again on the wire copy, never on that shared inventory.
    filtered = apply_llm_request_middleware(
        request, task_id=getattr(agent, "_current_task_id", "") or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=api_request_id, session_id=getattr(agent, "session_id", "") or "",
        platform=getattr(agent, "platform", "") or "", model=agent.model,
        provider=agent.provider, base_url=getattr(agent, "base_url", ""), api_mode=agent.api_mode,
    )
    return relay_llm.execute_current(
        filtered.payload, callback,
        name=str(getattr(agent, "provider", "") or "provider"), model_name=str(getattr(agent, "model", "") or ""),
        metadata={"api_mode": str(getattr(agent, "api_mode", "") or "chat_completions"),
                  "api_request_id": api_request_id, "call_role": "iteration_summary", "retry_count": retry_count},
        defer_logical_completion=True,
    )
