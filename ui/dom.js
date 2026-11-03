/* Small DOM primitives shared by the Artsy workspaces. */
(function(root) {
  'use strict';
  function reconcile(parent, children) {
    const wanted = new Set(children);
    for(let i=0;i<children.length;i++) {
      if(parent.children[i]!==children[i])parent.insertBefore(children[i],parent.children[i]||null);
    }
    for(const child of [...parent.children])if(!wanted.has(child))child.remove();
  }
  function setStatus(target, message) {
    target.textContent=message;
  }
  function openImage({src, alt='', dialog=null, returnFocus=null}) {
    const temporary=!dialog;
    if(temporary) {
      dialog=document.createElement('dialog');
      dialog.className='ui-image-dialog';
      dialog.setAttribute('aria-label',alt||'Image preview');
      const close=document.createElement('button');
      close.type='button';close.textContent='Close preview';close.onclick=()=>dialog.close();
      dialog.append(close,document.createElement('img'));
      dialog.onclose=()=>{dialog.remove();returnFocus?.focus();};
      document.body.append(dialog);
    }
    const img=dialog.querySelector('img');
    img.src=src;img.alt=alt||img.alt;
    dialog.showModal();
    return dialog;
  }
  root.ArtsyUI={reconcile,setStatus,openImage};
  if(typeof module!=='undefined' && module.exports)module.exports=root.ArtsyUI;
})(typeof window!=='undefined'?window:globalThis);
