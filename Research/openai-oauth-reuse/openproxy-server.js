import express from 'express';
import { fileURLToPath } from 'node:url';

const app = express();
app.use(express.json({ limit: '20mb' }));

const PORT = Number(process.env.PORT || 8082);
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-5.4';
const GENERIC_API_ERROR = Object.freeze({ error: { message: 'Internal server error' } });
const GENERIC_MODELS_ERROR = Object.freeze({ error: 'Internal server error' });

function safeConsoleError(message, error) {
  console.error(message, error);
}

function respondWithInternalJsonError(res, shape) {
  return res.status(500).json(shape);
}

function logServerError(context, error) {
  safeConsoleError(context, error);
}

function handleModelsError(res, error) {
  logServerError('Model listing failed', error);
  return respondWithInternalJsonError(res, GENERIC_MODELS_ERROR);
}

function handleMessagesError(res, error) {
  logServerError('Message proxy failed', error);
  return respondWithInternalJsonError(res, GENERIC_API_ERROR);
}

function mapModel(model) {
  if (!model) return OPENAI_MODEL;
  if (model.includes('sonnet') || model.includes('claude')) return OPENAI_MODEL;
  return model;
}

function getOpenAIToken(req) {
  return (
    process.env.OPENAI_OAUTH_TOKEN ||
    process.env.OPENAI_API_KEY ||
    (req.headers.authorization || '').replace(/^Bearer\s+/i, '') ||
    null
  );
}

function anthropicToOpenAIInput(body) {
  const input = [];

  for (const m of body.messages || []) {
    const role = m.role === 'assistant' ? 'assistant' : 'user';

    if (typeof m.content === 'string') {
      input.push({ role, content: [{ type: role === 'assistant' ? 'output_text' : 'input_text', text: m.content }] });
      continue;
    }

    for (const b of Array.isArray(m.content) ? m.content : []) {
      if (b?.type === 'text') {
        input.push({ role, content: [{ type: role === 'assistant' ? 'output_text' : 'input_text', text: b.text || '' }] });
      } else if (b?.type === 'tool_use' && role === 'assistant') {
        input.push({
          type: 'function_call',
          call_id: toOpenAICallId(b.id),
          name: b.name,
          arguments: JSON.stringify(b.input || {})
        });
      } else if (b?.type === 'tool_result') {
        input.push({
          type: 'function_call_output',
          call_id: toOpenAICallId(b.tool_use_id),
          output: typeof b.content === 'string' ? b.content : JSON.stringify(b.content ?? '')
        });
      }
    }
  }

  return input;
}

function anthropicToolsToOpenAITools(tools) {
  const normalizeSchema = (schema) => {
    const base = schema && typeof schema === 'object' ? schema : { type: 'object' };
    const properties = base.properties && typeof base.properties === 'object' ? base.properties : {};
    if (Object.keys(properties).length === 0) {
      // Codex /responses rejects object schemas without properties.
      // Keep tool permissive while satisfying validator.
      return {
        type: 'object',
        properties: { _arg: { type: 'string', description: 'Optional passthrough argument' } },
        additionalProperties: true
      };
    }
    return { ...base, properties };
  };

  return (tools || []).map((t) => ({
    type: 'function',
    name: t.name,
    description: t.description || '',
    parameters: normalizeSchema(t.input_schema)
  }));
}

function buildToolSchemaMap(tools) {
  const map = new Map();
  for (const t of tools || []) {
    map.set(t.name, t.input_schema || { type: 'object', properties: {} });
  }
  return map;
}

function sanitizeToolInputBySchema(name, input, toolSchemaMap) {
  const schema = toolSchemaMap.get(name);
  if (!schema || !schema.properties || typeof input !== 'object' || input == null) return input || {};

  const required = Array.isArray(schema.required) ? schema.required : [];
  const keysToKeep = required.length ? required : Object.keys(schema.properties);

  const out = {};
  for (const key of keysToKeep) {
    if (!(key in input)) continue;
    const propSchema = schema.properties[key] || {};
    const val = input[key];
    const t = propSchema?.type;
    if (t === 'string') out[key] = typeof val === 'string' ? val : String(val);
    else if (t === 'number' || t === 'integer') {
      const n = Number(val);
      if (!Number.isNaN(n)) out[key] = n;
    }
    else if (t === 'boolean') out[key] = Boolean(val);
    else out[key] = val;
  }

  for (const r of required) {
    if (!(r in out) && r in input) out[r] = input[r];
  }

  return out;
}

function logGuardrail(event, payload = {}) {
  try {
    const line = JSON.stringify({ ts: new Date().toISOString(), event, ...payload });
    console.log(`[guardrail] ${line}`);
  } catch {}
}

function parseToolArgs(args) {
  if (args == null) return {};
  if (typeof args === 'object') return args;
  if (typeof args !== 'string') return {};
  try { return JSON.parse(args); } catch { return {}; }
}

function toAnthropicToolUseId(callId) {
  const raw = String(callId || `call_${Date.now()}`);
  return raw.startsWith('toolu_') ? raw : `toolu_${raw.replace(/[^a-zA-Z0-9_-]/g, '')}`;
}

function toOpenAICallId(toolUseId) {
  const raw = String(toolUseId || '');
  return raw.startsWith('toolu_') ? raw.slice('toolu_'.length) : raw;
}

app.get('/', (_req, res) => {
  res.status(404).type('text/plain').send('Not found');
});

app.get('/health', (_req, res) => {
  res.json({ ok: true, provider: 'openai', base: OPENAI_BASE_URL });
});

app.get('/v1/models', async (req, res) => {
  try {
    const token = getOpenAIToken(req);
    if (!token) return res.status(401).json({ error: 'Missing OPENAI token' });

    const r = await fetch(`${OPENAI_BASE_URL}/models`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    const json = await r.json();
    res.status(r.status).json(json);
  } catch (e) {
    return handleModelsError(res, e);
  }
});

app.post('/v1/messages/count_tokens', (req, res) => {
  const body = req.body || {};
  const allText = [
    typeof body.system === 'string' ? body.system : '',
    ...(body.messages || []).map((m) =>
      typeof m.content === 'string'
        ? m.content
        : Array.isArray(m.content)
          ? m.content.filter((b) => b?.type === 'text').map((b) => b.text || '').join(' ')
          : ''
    )
  ].join(' ');

  const input_tokens = Math.ceil((allText || '').length / 4);
  return res.json({ input_tokens });
});

app.post('/v1/messages', async (req, res) => {
  try {
    const token = getOpenAIToken(req);
    if (!token) return res.status(401).json({ error: { message: 'Missing OPENAI token' } });

    const body = req.body || {};
    const stream = body.stream === true;
    const model = mapModel(body.model || OPENAI_MODEL);

    const input = anthropicToOpenAIInput(body);
    const toolSchemaMap = buildToolSchemaMap(body.tools);
    if (Array.isArray(body.tools) && body.tools.length) {
      logGuardrail('tool_schemas', {
        tools: body.tools.map((t) => ({
          name: t.name,
          required: t.input_schema?.required || [],
          props: Object.keys(t.input_schema?.properties || {})
        }))
      });
    }

    const oaBody = {
      model,
      store: false,
      stream: true,
      instructions: typeof body.system === 'string' ? body.system : 'You are a helpful coding assistant.',
      input,
      tools: anthropicToolsToOpenAITools(body.tools),
      tool_choice:
        body.tool_choice?.type === 'any'
          ? 'required'
          : body.tool_choice?.name
            ? { type: 'function', name: body.tool_choice.name }
            : 'auto'
    };

    logGuardrail('mapped_input', { input: oaBody.input.map((i) => ({ role: i.role, type: i.content?.[0]?.type })) });

    let upstream = await fetch(`${OPENAI_BASE_URL}/responses`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(oaBody)
    });

    if (!upstream.ok) {
      const txt = await upstream.text();
      if (upstream.status === 400 && txt.includes('invalid_value') && txt.includes('input[')) {
        logGuardrail('schema_mismatch_retry', { status: upstream.status, error: txt.slice(0, 300) });
        const retryBody = { ...oaBody, input: anthropicToOpenAIInput(body) };
        upstream = await fetch(`${OPENAI_BASE_URL}/responses`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
          },
          body: JSON.stringify(retryBody)
        });
        if (!upstream.ok) {
          const txt2 = await upstream.text();
          logGuardrail('schema_mismatch_retry_failed', { status: upstream.status, error: txt2.slice(0, 300) });
          return res.status(upstream.status).send(txt2);
        }
      } else {
        return res.status(upstream.status).send(txt);
      }
    }

    if (!upstream.body) {
      const txt = await upstream.text();
      return res.status(upstream.status).send(txt);
    }

    const reader = upstream.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    let msgId = `msg_${Date.now()}`;
    let inputTokens = 0;
    let outputTokens = 0;
    let aggregateText = '';
    let toolCalls = [];

    const parseSSEChunk = (chunk) => {
      const events = [];
      const normalized = chunk.replace(/\r\n/g, '\n');
      const frames = normalized.split('\n\n');
      const remainder = frames.pop() ?? '';

      for (const frame of frames) {
        const dataLines = [];
        for (const rawLine of frame.split('\n')) {
          if (!rawLine.startsWith('data:')) continue;
          dataLines.push(rawLine.slice(5).trimStart());
        }
        if (!dataLines.length) continue;
        const data = dataLines.join('\n').trim();
        if (!data || data === '[DONE]') continue;
        try { events.push(JSON.parse(data)); } catch {}
      }

      return { events, remainder };
    };

    const handleOpenAIEvent = (event) => {
      if (event?.type === 'response.output_text.delta') {
        const t = event.delta || '';
        aggregateText += t;
      }
      if (event?.type === 'response.output_item.done' && event?.item?.type === 'function_call') {
        toolCalls.push(event.item);
      }
      if (event?.type === 'response.completed') {
        inputTokens = event?.response?.usage?.input_tokens ?? inputTokens;
        outputTokens = event?.response?.usage?.output_tokens ?? outputTokens;
      }
    };

    if (!stream) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parsed = parseSSEChunk(buf);
        for (const event of parsed.events) handleOpenAIEvent(event);
        buf = parsed.remainder;
      }
      if (buf.trim()) {
        const parsed = parseSSEChunk(`${buf}\n\n`);
        for (const event of parsed.events) handleOpenAIEvent(event);
        buf = parsed.remainder;
      }

      const content = [];
      if (aggregateText) content.push({ type: 'text', text: aggregateText });
      for (const tc of toolCalls) {
        const parsedInput = sanitizeToolInputBySchema(tc.name, parseToolArgs(tc.arguments), toolSchemaMap);
        content.push({ type: 'tool_use', id: toAnthropicToolUseId(tc.call_id || tc.id), name: tc.name, input: parsedInput });
      }

      return res.json({
        id: msgId,
        type: 'message',
        role: 'assistant',
        model,
        content,
        stop_reason: toolCalls.length ? 'tool_use' : 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: outputTokens }
      });
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const sendEvent = (type, data) => {
      res.write(`event: ${type}\n`);
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    sendEvent('message_start', {
      type: 'message_start',
      message: {
        id: msgId,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 }
      }
    });
    let nextBlockIndex = 0;
    let textBlockIndex = -1;
    let sawToolCall = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const parsed = parseSSEChunk(buf);
      for (const event of parsed.events) {
        if (event?.type === 'response.output_text.delta') {
          if (textBlockIndex === -1) {
            textBlockIndex = nextBlockIndex++;
            sendEvent('content_block_start', {
              type: 'content_block_start',
              index: textBlockIndex,
              content_block: { type: 'text', text: '' }
            });
          }
          sendEvent('content_block_delta', {
            type: 'content_block_delta',
            index: textBlockIndex,
            delta: { type: 'text_delta', text: event.delta || '' }
          });
        }
        if (event?.type === 'response.output_item.done' && event?.item?.type === 'function_call') {
          sawToolCall = true;
          const parsedInput = sanitizeToolInputBySchema(event.item.name, parseToolArgs(event.item.arguments), toolSchemaMap);
          const idx = nextBlockIndex++;
          logGuardrail('tool_use_out', { name: event.item.name, call_id: event.item.call_id || event.item.id, anthropic_id: toAnthropicToolUseId(event.item.call_id || event.item.id), arg_type: typeof event.item.arguments, parsed_keys: Object.keys(parsedInput || {}) });
          sendEvent('content_block_start', {
            type: 'content_block_start',
            index: idx,
            content_block: {
              type: 'tool_use',
              id: toAnthropicToolUseId(event.item.call_id || event.item.id),
              name: event.item.name,
              input: {}
            }
          });
          sendEvent('content_block_delta', {
            type: 'content_block_delta',
            index: idx,
            delta: {
              type: 'input_json_delta',
              partial_json: JSON.stringify(parsedInput || {})
            }
          });
          sendEvent('content_block_stop', { type: 'content_block_stop', index: idx });
        }
        if (event?.type === 'response.completed') {
          inputTokens = event?.response?.usage?.input_tokens ?? inputTokens;
          outputTokens = event?.response?.usage?.output_tokens ?? outputTokens;
        }
      }
      buf = parsed.remainder;
    }

    if (buf.trim()) {
      const parsed = parseSSEChunk(`${buf}\n\n`);
      for (const event of parsed.events) {
        if (event?.type === 'response.output_text.delta') {
          if (textBlockIndex === -1) {
            textBlockIndex = nextBlockIndex++;
            sendEvent('content_block_start', {
              type: 'content_block_start',
              index: textBlockIndex,
              content_block: { type: 'text', text: '' }
            });
          }
          sendEvent('content_block_delta', {
            type: 'content_block_delta',
            index: textBlockIndex,
            delta: { type: 'text_delta', text: event.delta || '' }
          });
        }
        if (event?.type === 'response.output_item.done' && event?.item?.type === 'function_call') {
          sawToolCall = true;
          const parsedInput = sanitizeToolInputBySchema(event.item.name, parseToolArgs(event.item.arguments), toolSchemaMap);
          const idx = nextBlockIndex++;
          logGuardrail('tool_use_out', { name: event.item.name, call_id: event.item.call_id || event.item.id, anthropic_id: toAnthropicToolUseId(event.item.call_id || event.item.id), arg_type: typeof event.item.arguments, parsed_keys: Object.keys(parsedInput || {}) });
          sendEvent('content_block_start', {
            type: 'content_block_start',
            index: idx,
            content_block: {
              type: 'tool_use',
              id: toAnthropicToolUseId(event.item.call_id || event.item.id),
              name: event.item.name,
              input: {}
            }
          });
          sendEvent('content_block_delta', {
            type: 'content_block_delta',
            index: idx,
            delta: {
              type: 'input_json_delta',
              partial_json: JSON.stringify(parsedInput || {})
            }
          });
          sendEvent('content_block_stop', { type: 'content_block_stop', index: idx });
        }
        if (event?.type === 'response.completed') {
          inputTokens = event?.response?.usage?.input_tokens ?? inputTokens;
          outputTokens = event?.response?.usage?.output_tokens ?? outputTokens;
        }
      }
      buf = parsed.remainder;
    }

    if (textBlockIndex !== -1) {
      sendEvent('content_block_stop', { type: 'content_block_stop', index: textBlockIndex });
    }
    sendEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: sawToolCall ? 'tool_use' : 'end_turn', stop_sequence: null },
      usage: { output_tokens: outputTokens }
    });
    sendEvent('message_stop', { type: 'message_stop' });
    res.end();
  } catch (e) {
    return handleMessagesError(res, e);
  }
});

const isDirectRun = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];

if (isDirectRun) {
  app.listen(PORT, () => {
    console.log(`OpenAI OAuth -> Anthropic shim listening on http://localhost:${PORT}`);
  });
}

export { app };
