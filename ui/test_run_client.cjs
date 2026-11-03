const {test}=require('node:test');
const assert=require('node:assert/strict');
const {createRunClient}=require('./run-client.js');

function response(value, status=200){
  return {ok:status>=200&&status<300,status,json:async()=>value};
}
function deferred(){
  let resolve;
  const promise=new Promise(done=>{resolve=done;});
  return {promise,resolve};
}
async function until(check, timeout=300){
  const start=Date.now();
  while(!check()){
    if(Date.now()-start>timeout)throw new Error('Run client failed to advance');
    await new Promise(resolve=>setTimeout(resolve,2));
  }
}

test('retry after a transient failure, and keep polling the stopping state', async()=>{
  let attempts=0, retries=0, terminal=null;
  const updates=[];
  const client=createRunClient({
    interval:1,maxRetryDelay:4,
    request:async()=>{
      attempts++;
      if(attempts===1)throw new Error('Disconnected');
      return response({state:attempts===2?'stopping':'completed'});
    },
    onUpdate:run=>updates.push(run.state),
    onRetry:()=>{retries++;},
    onTerminal:run=>{terminal=run;}
  });
  client.start('example');
  await until(()=>terminal!==null);
  assert.equal(retries,1);
  assert.deepEqual(updates,['stopping','completed']);
  assert.equal(client.active,false);
});

test('ignore an older run response after switching to a new run', async()=>{
  const old=deferred();
  const updates=[];
  const client=createRunClient({
    interval:1,
    request:url=>url.endsWith('/old')?old.promise:Promise.resolve(response({state:'completed'})),
    onUpdate:(_,id)=>updates.push(id),
  });
  client.start('old');
  client.start('new');
  await until(()=>updates.includes('new'));
  old.resolve(response({state:'completed'}));
  await new Promise(resolve=>setTimeout(resolve,5));
  assert.deepEqual(updates,['new']);
});

test('cancel asks server to stop and keeps the run active until terminal', async()=>{
  let polled=0, cancelled=0, terminal=null;
  const client=createRunClient({
    interval:1,
    request:async(url,options={})=>{
      if(url.endsWith('/cancel')){
        cancelled++;
        return response({state:'stopping'});
      }
      polled++;
      return response({state:polled<3?'stopping':'failed'});
    },
    onTerminal:run=>{terminal=run;},
  });
  client.start('running');
  await client.cancel();
  assert.equal(cancelled,1);
  await until(()=>terminal!==null);
  assert.equal(terminal.state,'failed');
  assert.equal(client.active,false);
});

test('missing runs are reported once, without infinite retry', async()=>{
  let missing=0,retries=0;
  const client=createRunClient({
    interval:1,
    request:async()=>response({error:'Not found'},404),
    onMissing:()=>missing++,
    onRetry:()=>retries++,
  });
  client.start('missing');
  await until(()=>missing===1);
  assert.equal(retries,0);
  assert.equal(client.active,false);
});
