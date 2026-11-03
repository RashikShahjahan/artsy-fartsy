/* Shared, restartable run polling for Studio and consumer Artsy. */
(function(root) {
  'use strict';
  const ACTIVE_STATES = new Set(['queued', 'running', 'stopping']);
  function createRunClient({endpoint='/api/runs', interval=1500, maxRetryDelay=10000,
      request=(url, options)=>fetch(url, options), onUpdate=()=>{}, onTerminal=()=>{},
      onRetry=()=>{}, onMissing=()=>{}}={}) {
    let id=null, epoch=0, timer=null, controller=null, failures=0;
    const isCurrent = (version, runId) => epoch===version && id===runId;
    function stop() {
      epoch++;
      clearTimeout(timer);
      timer=null;
      controller?.abort();
      controller=null;
      id=null;
      failures=0;
    }
    function schedule(version, delay) {
      if(!isCurrent(version,id))return;
      clearTimeout(timer);
      timer=setTimeout(()=>poll(version),delay);
    }
    async function poll(version) {
      const runId=id;
      if(!runId || !isCurrent(version,runId))return;
      controller=new AbortController();
      try {
        const response=await request(endpoint+'/'+encodeURIComponent(runId),
          {cache:'no-store',signal:controller.signal});
        const data=await response.json();
        if(!isCurrent(version,runId))return;
        if(!response.ok) {
          const error=new Error(data?.error || 'Could not read run ('+response.status+')');
          error.status=response.status;
          throw error;
        }
        failures=0;
        controller=null;
        onUpdate(data,runId);
        if(!isCurrent(version,runId))return;
        if(ACTIVE_STATES.has(data.state))schedule(version,interval);
        else {
          stop();
          onTerminal(data,runId);
        }
      } catch(error) {
        if(!isCurrent(version,runId) || error.name==='AbortError')return;
        controller=null;
        if(error.status===404) {
          stop();
          onMissing(runId,error);
          return;
        }
        failures++;
        const delay=Math.min(maxRetryDelay,interval*Math.pow(2,Math.min(8,failures-1)));
        onRetry(error,{id:runId,failures,delay});
        if(isCurrent(version,runId))schedule(version,delay);
      }
    }
    function start(runId) {
      if(!runId)throw new Error('A run ID is required');
      stop();
      id=runId;
      const version=epoch;
      void poll(version);
    }
    async function cancel() {
      const runId=id, version=epoch;
      if(!runId)return false;
      const response=await request(endpoint+'/'+encodeURIComponent(runId)+'/cancel',
        {method:'POST'});
      let data;
      try { data=await response.json(); } catch { data={}; }
      if(!response.ok)throw new Error(data.error || 'Could not cancel run');
      if(isCurrent(version,runId)) {
        clearTimeout(timer);
        if(controller)controller.abort();
        void poll(version);
      }
      return true;
    }
    return {start,stop,cancel,get id(){return id;},get active(){return id!==null;}};
  }
  root.ArtsyRunClient={createRunClient,ACTIVE_STATES};
  if(typeof module!=='undefined' && module.exports)module.exports=root.ArtsyRunClient;
})(typeof window!=='undefined' ? window : globalThis);
