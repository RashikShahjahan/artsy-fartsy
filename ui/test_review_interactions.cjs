const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function harness(script, payload, {record=null}={}){
  class Element {
    constructor(tagName='div',id=''){
      this.tagName=tagName.toUpperCase();this.id=id;this.children=[];this.dataset={};
      this.attributes={};this.value='';this.checked=false;this.hidden=false;
      this.disabled=false;this.style={};this.parentNode=null;this._text='';
      this.classList={
        toggle:()=>{},add:()=>{},remove:()=>{},
      };
    }
    get textContent(){return this._text;}
    set textContent(value){this._text=String(value);this.replaceChildren();}
    get childNodes(){return this.children;}
    get firstElementChild(){return this.children[0]||null;}
    append(...nodes){for(const node of nodes){node.remove?.();node.parentNode=this;this.children.push(node);}}
    remove(){if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(child=>child!==this);this.parentNode=null;}
    insertBefore(node,anchor){
      node.remove?.();
      const position=this.children.indexOf(anchor);
      node.parentNode=this;
      this.children.splice(position<0?this.children.length:position,0,node);
    }
    replaceChildren(...nodes){
      for(const child of this.children)child.parentNode=null;
      this.children=[];
      this.append(...nodes);
    }
    setAttribute(name,value){this.attributes[name]=String(value);}
    getAttribute(name){return this.attributes[name]||null;}
    querySelectorAll(selector){
      const found=[];
      function walk(node){
        for(const child of node.children){
          if(selector==='[data-review-decision]' && child.dataset.reviewDecision)found.push(child);
          walk(child);
        }
      }
      walk(this);return found;
    }
    querySelector(tag){return this.children.find(child=>child.tagName===tag.toUpperCase())||null;}
    focus(){}showModal(){this.open=true;}close(){this.open=false;}
  }
  const ids=new Map();
  for(const id of ['review-data','cards','search','slice','filter','reviewer','save','final','status',
      'overview','items','detail','automatic','decision','export','start','review','results',
      'previous','next','unranked','reveal','prompt','position','images','zoom','close-zoom','score-table']){
    ids.set(id,new Element('div',id));
  }
  ids.get('review-data').textContent=JSON.stringify(payload);
  ids.get('zoom').append(new Element('img'));
  function find(id){
    if(ids.has(id))return ids.get(id);
    const visited=new Set();
    function scan(node){
      if(visited.has(node))return null;
      visited.add(node);
      if(node.id===id)return node;
      for(const child of node.children){const match=scan(child);if(match)return match;}
      return null;
    }
    for(const node of ids.values()){const match=scan(node);if(match)return match;}
    return null;
  }
  const document={
    getElementById:find,createElement:tag=>new Element(tag),
    body:new Element('body')
  };
  const pending=[];
  const fetch=(url,options={})=>{
    if(record && url.includes('/api/record?')){
      return Promise.resolve({ok:true,json:async()=>record});
    }
    if(!options.method && payload.cases){
      return Promise.resolve({ok:true,json:async()=>payload});
    }
    return new Promise(resolve=>pending.push({url,options,resolve}));
  };
  const context=vm.createContext({
    document,fetch,window:{addEventListener(){},ArtsyUI:{
      setStatus(element,value){element.textContent=value;},
      reconcile(element,nodes){element.replaceChildren(...nodes);},
      openImage(){},
    }},
    localStorage:{getItem(){return null;},setItem(){}},
    navigator:{clipboard:{writeText:async()=>{}}},
  });
  vm.runInContext(fs.readFileSync(__dirname+'/'+script,'utf8'),context);
  return {ids,pending,context,find};
}
function accept(request,value={saved:true}){
  request.resolve({ok:true,json:async()=>value});
}
async function flush(){await new Promise(resolve=>setImmediate(resolve));}

test('prompt review acknowledges only the submitted notes revision',async()=>{
  const payload={
    items:[{id:'prompt-a',prompt:'A circle',benchmarkSlice:'drawing',
      difficulty:'easy',skills:[],seed:42,kind:'drawing',benchmarkSource:'test'}],
    review:{decisions:{}},sha256:'test',scope:'all',basePath:'/review'
  };
  const h=harness('prompt-review.js',payload);
  const notes=h.ids.get('cards').children[0].children.find(child=>child.tagName==='TEXTAREA');
  notes.value='original';notes.oninput();
  const saving=h.ids.get('save').onclick();
  assert.equal(JSON.parse(h.pending[0].options.body).decisions['prompt-a'].notes,'original');
  notes.value='edited while saving';notes.oninput();
  accept(h.pending.shift());
  await saving;
  assert.equal(vm.runInContext('changed',h.context),true);
  assert.match(h.ids.get('status').textContent,/newer edits are still unsaved/);
  assert.equal(h.ids.get('cards').children[0].children.find(child=>child.tagName==='TEXTAREA'),notes);
});

test('candidate decisions preserve loaded detail nodes, and saves preserve newer notes',async()=>{
  const payload={
    items:[{id:'candidate-a',prompt:'A fox',family:'drawing',reason:'',status:'review',score:null}],
    review:{decisions:{}},sha256:'test',basePath:'/candidates',reviewPath:'manual-review.json'
  };
  const record={record:{id:'candidate-a',spec:{prompt:'A fox'},status:'review',reason:'',trace:[]},
    images:[],example:null,conversionError:'not convertible'};
  const h=harness('candidate-review.js',payload,{record});
  h.ids.get('items').children[0].onclick();
  await flush();
  const detail=h.ids.get('detail');
  const heading=detail.children[0],bar=detail.children[2];
  bar.children[0].onclick();
  assert.equal(detail.children[0],heading,'decision must not recreate candidate details');
  const notes=detail.children.find(child=>child.tagName==='TEXTAREA');
  notes.value='first';notes.oninput();
  h.ids.get('reviewer').value='Reviewer';h.ids.get('reviewer').oninput();
  const saving=h.ids.get('save').onclick();
  assert.equal(JSON.parse(h.pending[0].options.body).decisions['candidate-a'].notes,'first');
  notes.value='second';notes.oninput();
  accept(h.pending.shift());
  await saving;
  assert.equal(vm.runInContext('changed',h.context),true);
  assert.match(h.ids.get('status').textContent,/newer edits are still unsaved/);
});

test('saving rank A after navigating to B never marks B ranked locally',async()=>{
  const payload={sealed:false,cases:[
    {id:'caseA',prompt:'Draw A',available:[true,true,true],ranks:null},
    {id:'caseB',prompt:'Draw B',available:[true,true,true],ranks:null}
  ]};
  const h=harness('preferences.js',payload);
  h.ids.get('reviewer').value='Reviewer';
  await h.ids.get('start').onsubmit({preventDefault(){}});
  for(let i=0;i<3;i++){
    const select=h.find('rank-'+i);
    select.value=String(i+1);
    select.onchange();
  }
  const saving=h.ids.get('save').onclick();
  assert.equal(JSON.parse(h.pending[0].options.body).case,'caseA');
  h.ids.get('next').onclick();
  accept(h.pending.shift());
  await saving;
  const state=vm.runInContext('({packet,index})',h.context);
  assert.deepEqual(Array.from(state.packet.cases[0].ranks),[1,2,3]);
  assert.equal(state.packet.cases[1].ranks,null);
  assert.equal(state.index,1);
});
