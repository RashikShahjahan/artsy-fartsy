const {test} = require('node:test');
const assert = require('node:assert/strict');
const {Activity, stepsFor} = require('./activity.js');
const traceApi = require('./trace.js');

function node(tag = 'div') {
  return {tag, children: [], textContent: '', hidden: true, scrollTop: 0, scrollHeight: 0, clientHeight: 300,
    append(...children) { this.children.push(...children); children.forEach(child => child.parent = this); },
    replaceChildren(...children) { this.children = []; this.append(...children); },
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }};
}
function find(parent, tag) { return parent.children.flatMap(child => [child, ...find(child, tag)]).filter(child => child.tag === tag); }
const run = (overrides = {}) => ({id: 'a'.repeat(32), state: 'running', execution_mode: 'approval',
  progress: {phase: 'awaiting_approval', turn: 1, max_turns: 10, steps: [
    {id: 'turn-1', kind: 'model', title: 'Agent turn 1', status: 'completed', output: '<script>text only</script>'},
    {id: 'action-1', kind: 'tool', title: 'Write drawing code', status: 'pending', command: 'draw'},
  ]}, ...overrides});

test('response summary titles replace numbered turns without repeating the heading', () => {
  const current = run();
  current.progress.steps[0].output = '**Designing wrecking ball illustration**\n\nI’ll sketch the crane first.';
  const step = stepsFor(current)[0];
  assert.equal(step.title, 'Designing wrecking ball illustration');
  assert.equal(step.output, 'I’ll sketch the crane first.');
  assert.equal(current.progress.steps[0].title, 'Agent turn 1');
  current.progress.steps[0].output = '### Adjusting the shadow\n\nMoving it below the ball.';
  assert.equal(stepsFor(current)[0].title, 'Adjusting the shadow');
  current.progress.steps[0].output = '';
  current.progress.steps[0].status = 'running';
  assert.equal(stepsFor(current)[0].title, 'Thinking…');
  current.progress.steps[0].status = 'completed';
  assert.equal(stepsFor(current)[0].title, 'Drawing update');
  const saved = stepsFor({state: 'completed', trace: [{samples: [{messages: [
    {role: 'assistant', content: '**Refining the leaves**\n\nAdding smaller shapes.'},
  ]}]}]}, traceApi);
  assert.equal(saved[0].title, 'Refining the leaves');
  assert.equal(saved[0].output, 'Adding smaller shapes.');
});

test('submission is only complete after final validation and image availability', () => {
  assert.equal(stepsFor(run({state: 'completed', result: {outcome: 'submitted'}})).at(-1).status, 'failed');
  assert.equal(stepsFor(run({state: 'completed', result: {outcome: 'submitted'}, artifacts: ['image.png']})).at(-1).title, 'Drawing ready');
  const refused = stepsFor(run({state: 'completed', result: {outcome: 'refused', reason: 'Try another idea'}}));
  assert.equal(refused.at(-1).status, 'refused');
  assert.equal(refused.at(-1).output, 'Try another idea');
  assert.equal(refused[1].status, 'interrupted');
});

test('old saved traces recover every recorded action and preview', () => {
  const steps = stepsFor({state: 'completed', trace: [{samples: [{messages: [
    {role: 'assistant', content: 'Rendering', tool_calls: [{id: 'one', function: 'bash', arguments: {command: '/opt/artsy/render.py --preview'}}]},
    {role: 'tool', tool_call_id: 'one', content: [{type: 'text', text: 'Return code: 0'}, {type: 'image', image: 'data:image/png;base64,AAAA'}]},
  ]}]}]}, traceApi);
  assert.equal(steps[0].title, 'Render preview');
  assert.equal(steps[0].status, 'completed');
  assert.equal(steps[0].previews.length, 1);
});

test('polling preserves expanded details and offers approval only for pending steps', async () => {
  global.document = {createElement: node};
  const panel = node(), list = node('ol'), status = node();
  const approved = [];
  const activity = new Activity(panel, list, status, async id => { approved.push(id); return true; });
  activity.render(run());
  assert.equal(status.textContent, 'Your approval is needed');
  assert.equal(list.children.length, 2);
  assert.equal(find(list, 'p')[0].textContent, '<script>text only</script>');
  let details = find(list, 'details')[0]; details.open = true;
  activity.render(run());
  assert.equal(find(list, 'details')[0], details);
  const button = find(list, 'button')[0];
  await button.onclick();
  assert.deepEqual(approved, ['action-1']);
  assert.equal(button.disabled, true);
  const next = run(); next.progress.phase = 'drawing'; next.progress.steps[1].status = 'running';
  activity.render(next);
  assert.equal(find(list, 'details')[0].open, true);
  assert.equal(find(list, 'button').length, 0);
  const auto = run({id: 'b'.repeat(32), execution_mode: 'auto'});
  activity.render(auto);
  assert.equal(find(list, 'button').length, 0);
  activity.reset(); assert.equal(panel.hidden, true); assert.equal(list.children.length, 0);
});

test('failed actions open their output and previews use only approved image paths', () => {
  global.document = {createElement: node};
  const panel = node(), list = node(), status = node();
  const activity = new Activity(panel, list, status);
  const current = run();
  activity.render(current);
  assert.equal(find(list, 'details')[0].open, false);
  current.progress.steps[1] = {id: 'action-1', kind: 'tool', title: 'Render preview',
    status: 'failed', output: 'SyntaxError', previews: ['preview-1-0.png', 'https://untrusted.example/image.png', '../secret']};
  activity.render(current);
  assert.equal(find(list, 'details')[0].open, true);
  assert.equal(find(list, 'img').length, 1);
  assert.equal(find(list, 'img')[0].src, `/api/runs/${current.id}/preview-1-0.png`);
});
