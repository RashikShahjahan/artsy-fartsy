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
function actionDetails(parent) { return find(parent, 'div').find(child => child.className === 'activity-details'); }
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

test('action details appear directly and polling offers approval only for pending steps', async () => {
  global.document = {createElement: node};
  const panel = node(), list = node('ol'), status = node();
  const approved = [];
  const activity = new Activity(panel, list, status, async id => { approved.push(id); return true; });
  activity.render(run());
  assert.equal(status.textContent, 'Your approval is needed');
  assert.equal(list.children.length, 2);
  assert.equal(find(list, 'p')[0].textContent, '<script>text only</script>');
  const details = actionDetails(list);
  assert.equal(find(details, 'p')[0].textContent, 'Create or update the instructions that make your drawing.');
  assert.equal(find(list, 'pre').length, 0);
  assert.equal(find(list, 'details').length, 0);
  assert.equal(find(list, 'summary').length, 0);
  assert.equal(find(list.children[1], 'strong').length, 0);
  activity.render(run());
  assert.equal(actionDetails(list), details);
  const button = find(list, 'button')[0];
  await button.onclick();
  assert.deepEqual(approved, ['action-1']);
  assert.equal(button.disabled, true);
  const next = run(); next.progress.phase = 'drawing'; next.progress.steps[1].status = 'running';
  activity.render(next);
  assert.equal(find(actionDetails(list), 'p')[0].textContent, 'Create or update the instructions that make your drawing.');
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
  assert.equal(find(actionDetails(list), 'p')[0].textContent, 'Create or update the instructions that make your drawing.');
  current.progress.steps[1] = {id: 'action-1', kind: 'tool', title: 'Render preview',
    status: 'failed', output: 'SyntaxError', previews: ['preview-1-0.png', 'https://untrusted.example/image.png', '../secret']};
  activity.render(current);
  assert.match(find(actionDetails(list), 'p')[1].textContent, /couldn’t finish/);
  assert.equal(find(list, 'pre').length, 0);
  assert.equal(find(list, 'img').length, 1);
  assert.equal(find(list, 'img')[0].src, `/api/runs/${current.id}/preview-1-0.png`);
});

test('all recorded actions and previews remain visible during auto mode and later thinking', () => {
  global.document = {createElement: node};
  const list = node('ol');
  const activity = new Activity(node(), list, node());
  const current = run({execution_mode: 'auto'});
  current.progress.steps[1].status = 'completed';
  current.progress.steps[1].previews = ['preview-1-0.png'];
  activity.render(current);
  assert.deepEqual(list.children.map(row => row.hidden), [false, false]);
  assert.equal(find(list, 'button').length, 0);
  current.progress.steps.push({id: 'turn-2', kind: 'model', title: 'Agent turn 2', status: 'running'});
  activity.render(current);
  assert.deepEqual(list.children.map(row => row.hidden), [false, false, true]);
  assert.equal(find(list.children[1], 'img')[0].src, `/api/runs/${current.id}/preview-1-0.png`);
  current.progress.steps[2].output = 'Adding the finishing touches.';
  activity.render(current);
  assert.deepEqual(list.children.map(row => row.hidden), [false, false, false]);
  current.progress.steps.push({id: 'action-2', kind: 'tool', title: 'Render preview', status: 'completed', previews: ['preview-3-0.png']});
  activity.render(current);
  assert.equal(find(list, 'img').length, 2);
  assert.equal(list.children.every(row => !row.hidden), true);
  assert.equal(find(list, 'button').length, 0);
  current.state = 'completed'; current.result = {outcome: 'submitted'}; current.artifacts = ['image.png'];
  activity.render(current);
  assert.equal(list.children.every(row => !row.hidden), true);
});

test('action descriptions explain shell effects without showing commands or tool logs', () => {
  assert.match(actionDescription({command: "cat > /workspace/program.py <<'PY'\nsecret code\nPY\n/opt/artsy/render.py --preview"}), /instructions.*Render a preview/);
  assert.match(actionDescription({command: 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}), /Submit your drawing/);
  assert.match(actionDescription({command: 'sed -i s/red/blue/ program.py'}), /Adjust the instructions/);
  assert.match(actionDescription({command: 'cat program.py'}), /Inspect the drawing files/);
  assert.match(actionDescription({command: 'python program.py'}), /Run the drawing instructions/);
  assert.equal(actionDescription({command: 'unknown --secret-value'}).includes('secret'), false);
});

test('image loading follows new steps unless the user scrolls back', () => {
  global.document = {createElement: node};
  const list = node('ol');
  const activity = new Activity(node(), list, node());
  const current = run(); current.progress.steps[1].previews = ['preview-1-0.png'];
  activity.render(current);
  const image = find(list, 'img')[0];
  list.scrollHeight = 900; image.onload();
  assert.equal(list.scrollTop, 900);
  list.scrollTop = 0; list.onscroll(); image.onload();
  assert.equal(list.scrollTop, 0);
});
