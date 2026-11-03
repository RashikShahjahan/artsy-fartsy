const {test} = require('node:test');
const assert = require('node:assert/strict');
const {Activity, stepsFor, actionDescription} = require('./activity.js');
const traceApi = require('./trace.js');

function node(tag = 'div') {
  return {tag, children: [], textContent: '', hidden: true, scrollTop: 0, scrollHeight: 0, clientHeight: 300,
    append(...children) { this.children.push(...children); children.forEach(child => child.parent = this); },
    replaceChildren(...children) { this.children = []; this.append(...children); },
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }};
}
function find(parent, tag) { return parent.children.flatMap(child => [child, ...find(child, tag)]).filter(child => child.tag === tag); }
function title(parent) { return find(parent, 'strong')[0].textContent; }
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

test('only the current action appears and polling offers approval only for pending steps', async () => {
  global.document = {createElement: node};
  const panel = node(), list = node('ol'), status = node();
  const approved = [];
  const activity = new Activity(panel, list, status, async id => { approved.push(id); return true; });
  activity.render(run());
  assert.equal(status.textContent, 'Your approval is needed');
  assert.equal(list.children.length, 1);
  assert.equal(title(list), 'Create your drawing');
  const heading = find(list, 'strong')[0];
  assert.equal(find(list, 'p').length, 0);
  assert.equal(find(list, 'pre').length, 0);
  assert.equal(find(list, 'details').length, 0);
  assert.equal(find(list, 'summary').length, 0);
  activity.render(run());
  assert.equal(find(list, 'strong')[0], heading);
  const button = find(list, 'button')[0];
  await button.onclick();
  assert.deepEqual(approved, ['action-1']);
  assert.equal(button.disabled, true);
  const next = run(); next.progress.phase = 'drawing'; next.progress.steps[1].status = 'running';
  activity.render(next);
  assert.equal(title(list), 'Creating your drawing…');
  assert.equal(find(list, 'button').length, 0);
  const auto = run({id: 'b'.repeat(32), execution_mode: 'auto'});
  activity.render(auto);
  assert.equal(find(list, 'button').length, 0);
  activity.reset(); assert.equal(panel.hidden, true); assert.equal(list.children.length, 0);
});

test('failed action output stays visible and previews use only approved image paths', () => {
  global.document = {createElement: node};
  const panel = node(), list = node(), status = node();
  const activity = new Activity(panel, list, status);
  const current = run();
  activity.render(current);
  assert.equal(title(list), 'Create your drawing');
  current.progress.steps[1] = {id: 'action-1', kind: 'tool', title: 'Render preview',
    status: 'failed', output: 'SyntaxError', previews: ['preview-1-0.png', 'https://untrusted.example/image.png', '../secret']};
  activity.render(current);
  assert.match(title(list), /syntax error/);
  assert.equal(find(list, 'pre').length, 0);
  assert.equal(find(list, 'img').length, 1);
  assert.equal(find(list, 'img')[0].src, `/api/runs/${current.id}/preview-1-0.png`);
});

test('assistant and action appear separately with friendly results always visible', () => {
  global.document = {createElement: node};
  const list = node('ol'), activity = new Activity(node(), list, node());
  const current = run();
  current.progress.steps[0].turn = 1;
  current.progress.steps[0].output = 'I’ll add the leaves next.';
  Object.assign(current.progress.steps[1], {turn: 1, tool: 'update_drawing',
    arguments: {source: '<drawing source>'}, output: 'Return code: 0', seconds: 1.25});
  activity.render(current);
  assert.equal(title(list), 'Refine your drawing');
  assert.equal(list.children.length, 1);
  assert.equal(find(list, 'details').length, 0);
  assert.equal(find(list, 'summary').length, 0);
  assert.equal(find(list, 'pre').length, 0);
  assert.equal(find(list, 'p').some(item => item.textContent.includes('<drawing source>')), false);
  current.progress.steps[1].status = 'completed';
  activity.render(current);
  assert.equal(title(list), 'The drawing code was saved and is ready to preview.');
  assert.equal(find(list, 'p').length, 0);
});

test('each new stage replaces the previous action and preview in both modes', () => {
  for (const execution_mode of ['auto', 'approval']) {
    global.document = {createElement: node};
    const list = node('ol');
    const activity = new Activity(node(), list, node());
    const current = run({execution_mode});
    current.progress.steps[1].status = 'completed';
    current.progress.steps[1].previews = ['preview-1-0.png'];
    activity.render(current);
    assert.equal(list.children.length, 1);
    assert.equal(title(list), 'Drawing created');
    assert.equal(find(list, 'button').length, 0);
    current.progress.steps.push({id: 'turn-2', kind: 'model', title: 'Agent turn 2', status: 'running'});
    activity.render(current);
    assert.equal(list.children.length, 1);
    assert.equal(list.children[0].hidden, false);
    assert.equal(title(list), 'Drawing created');
    assert.equal(find(list, 'img').length, 1);
    current.progress.steps[2].output = 'Adding the finishing touches.';
    activity.render(current);
    assert.deepEqual(list.children.map(row => row.hidden), [false]);
    assert.equal(find(list, 'p')[0].textContent, 'Adding the finishing touches.');
    current.progress.steps[2].status = 'completed';
    current.progress.steps.push({id: 'action-2', kind: 'tool', title: 'Render preview', status: 'completed', previews: ['preview-3-0.png']});
    activity.render(current);
    assert.equal(list.children.length, 1);
    assert.equal(find(list, 'img').length, 1);
    assert.equal(find(list, 'img')[0].src, `/api/runs/${current.id}/preview-3-0.png`);
    assert.equal(list.children.every(row => !row.hidden), true);
    assert.equal(find(list, 'button').length, 0);
    current.state = 'completed'; current.result = {outcome: 'submitted'}; current.artifacts = ['image.png'];
    activity.render(current);
    assert.equal(list.children.length, 2);
    assert.equal(find(list, 'img').length, 0);
    assert.equal(list.children.every(row => !row.hidden), true);
    assert.equal(title(list.children[0]), 'Drawing update');
    for (let index = 0; index < 4; index++) find(list, 'button').at(-1).onclick();
    assert.equal(title(list.children[0]), 'Drawing ready');
    assert.equal(list.children.length, 2);
  }
});

test('initial thinking stays visible and empty model responses preserve actions between polls', () => {
  global.document = {createElement: node};
  const list = node('ol');
  const activity = new Activity(node(), list, node());
  const current = run({execution_mode: 'auto', progress: {phase: 'thinking', steps: []}});
  activity.render(current);
  assert.equal(list.children[0].hidden, false);
  assert.equal(title(list), 'Getting ready to draw');
  current.progress.steps.push({id: 'turn-1', kind: 'model', title: 'Agent turn 1', status: 'running'});
  activity.render(current);
  assert.equal(list.children[0].hidden, false);
  assert.equal(title(list), 'Thinking…');
  current.progress.steps[0].status = 'completed';
  current.progress.steps.push(
    {id: 'action-1', tool: 'update_drawing', status: 'completed'},
    {id: 'turn-2', kind: 'model', title: 'Agent turn 2', status: 'running'});
  activity.render(current);
  assert.equal(list.children.length, 1);
  assert.equal(list.children[0].hidden, false);
  assert.equal(title(list), 'The drawing code was saved and is ready to preview.');
  current.progress.steps[2].status = 'completed';
  activity.render(current);
  assert.equal(title(list), 'The drawing code was saved and is ready to preview.');
});

test('shell actions use short, status-aware drawing labels', () => {
  assert.equal(actionDescription({command: "cat > /workspace/program.py <<'PY'\nsecret code\nPY\n/opt/artsy/render.py --preview"}), 'Checking the preview…');
  assert.equal(actionDescription({command: 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}), 'Finishing your drawing…');
  assert.equal(actionDescription({command: 'sed -i s/red/blue/ program.py'}), 'Refining your drawing…');
  assert.equal(actionDescription({command: 'cat program.py'}), 'Reviewing your drawing…');
  assert.equal(actionDescription({command: 'python program.py'}), 'Checking your drawing…');
  assert.equal(actionDescription({kind: 'write', status: 'pending'}), 'Create your drawing');
  assert.equal(actionDescription({kind: 'render', status: 'completed'}), 'Preview checked');
  assert.equal(actionDescription({command: 'unknown --secret-value'}).includes('secret'), false);
});

test('named drawing tools drive consumer labels without inspecting source or commands', () => {
  assert.equal(actionDescription({tool: 'update_drawing', kind: 'write', status: 'running'}), 'Refining your drawing…');
  assert.equal(actionDescription({tool: 'preview_drawing', status: 'pending'}), 'Check the preview');
  assert.equal(actionDescription({tool: 'read_drawing', status: 'completed'}), 'Drawing reviewed');
  assert.equal(actionDescription({tool: 'refuse_drawing', status: 'completed'}), 'Request declined');
  const saved = stepsFor({state: 'completed', trace: [{samples: [{messages: [
    {role: 'assistant', tool_calls: [{id: 'one', function: 'preview_drawing', arguments: {}}]},
    {role: 'tool', tool_call_id: 'one', content: 'Return code: 0'},
  ]}]}]}, traceApi);
  assert.equal(actionDescription(saved[0]), 'Preview checked');
});

test('final checks replace earlier steps without changing the recorded history', () => {
  global.document = {createElement: node};
  const list = node('ol');
  const activity = new Activity(node(), list, node());
  const current = run();
  current.progress.phase = 'rendering';
  activity.render(current);
  activity.render(current);
  assert.equal(list.children.length, 1);
  assert.equal(title(list), 'Checking the final drawing');
  assert.equal(current.progress.steps.length, 2);
  current.state = 'completed';
  activity.render(current);
  activity.render(current);
  assert.equal(list.children.length, 2);
  assert.equal(current.progress.steps.length, 2);
});

test('polling preserves the current stage scroll position and new stages start at the top', () => {
  global.document = {createElement: node};
  const list = node('ol');
  const activity = new Activity(node(), list, node());
  const current = run(); current.progress.steps[1].previews = ['preview-1-0.png'];
  activity.render(current);
  list.scrollHeight = 900; list.scrollTop = 200;
  activity.render(current);
  assert.equal(list.scrollTop, 200);
  current.progress.steps[1].status = 'completed';
  current.progress.steps.push({id: 'turn-2', kind: 'model', title: 'Agent turn 2',
    status: 'running', output: 'Adding the leaves.'});
  activity.render(current);
  assert.equal(list.scrollTop, 0);
});
