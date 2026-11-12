"""Iteration-limit continuations must retain per-request middleware restrictions."""
from copy import deepcopy
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agent import chat_completion_helpers as helpers
# Match native startup: bootstrap before the per-test home I/O guard.
importlib.import_module('agent.conversation_loop')
from hermes_cli.middleware import RequestMiddlewareResult


@pytest.mark.parametrize('mode', ['chat_completions', 'anthropic_messages', 'codex_responses'])
def test_iteration_summary_retains_request_policy(mode):
    cached = [{'type': 'function', 'function': {'name': name}} for name in ('room_read', 'room_post', 'host_tool')]
    original = deepcopy(cached)
    built = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'original'}], 'tools': cached}
    agent = SimpleNamespace(
        api_mode=mode, provider='custom', model='fixture', base_url='http://127.0.0.1:1',
        session_id='session', platform='fixture', _current_turn_id='turn', _current_task_id='task',
        _build_api_kwargs=lambda _: deepcopy(built),
        _interruptible_api_call=lambda request: request,
    )
    transport = SimpleNamespace(
        build_kwargs=lambda **kw: deepcopy(built),
        normalize_response=lambda response, **kw: SimpleNamespace(content='finished', tool_calls=[]))
    agent._get_transport = lambda: transport
    agent.max_tokens = 100
    agent.reasoning_config = {}
    agent._is_anthropic_oauth = False
    agent._anthropic_preserve_dots = lambda: False
    seen = []

    def restrict(request, **context):
        seen.append(context)
        value = deepcopy(request)
        value['tools'] = [t for t in value.get('tools', []) if t['function']['name'] != 'host_tool']
        return RequestMiddlewareResult(value, request, changed=True)

    wire = []
    def send(request):
        wire.append(request)
        return request
    agent._interruptible_api_call = send
    def relay(request, callback, **kw):
        return callback(request)

    with patch('hermes_cli.middleware.apply_llm_request_middleware', side_effect=restrict), \
         patch('agent.relay_llm.execute_current', side_effect=relay), \
         patch.object(helpers, 'sanitize_outbound_kwargs'), \
         patch.object(helpers, '_merge_nous_portal_messages_extra_body', side_effect=lambda a,k,m:k):
        builder = helpers._SUMMARY_ATTEMPT_BUILDERS.get(mode, helpers._chat_summary_attempt)
        assert builder(agent, built['messages'], 'summary-id')(0) == 'finished'
    assert len(seen) == 1
    assert seen[0]['session_id'] == 'session'
    assert seen[0]['turn_id'] == 'turn'
    assert seen[0]['task_id'] == 'task'
    assert seen[0]['api_mode'] == mode
    assert all(t['function']['name'] != 'host_tool' for t in wire[0].get('tools', []))
    assert cached == original
