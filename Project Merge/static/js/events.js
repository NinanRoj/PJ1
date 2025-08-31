console.log('loaded: events.js');
import { addFiles, removeFile, resetFiles, getFiles } from './state.js';
import { render, openPreview } from './ui.js';

const input     = document.getElementById('pdfInput');
const chooseBtn = document.getElementById('chooseBtn');
const dropzone  = document.getElementById('dropzone');
const addBtn    = document.getElementById('addBtn');
const pickBtn   = document.getElementById('pickBtn');
const resetBtn  = document.getElementById('resetBtn');
const checkBtn  = document.getElementById('checkBtn');

export function initEvents(){
  function rerender(){
    render(
      (id)=>{ removeFile(id); rerender(); },
      (id)=>{ const f = getFiles().find(x=>x.id===id); if (f) openPreview(f.file); }
    );
  }

  const attachFiles = (list) => { addFiles(list); rerender(); };

  // ปุ่มเลือกไฟล์
  chooseBtn.addEventListener('click', ()=> input.click());
  addBtn?.addEventListener('click', ()=> input.click());
  pickBtn?.addEventListener('click', ()=> input.click());

  // เลือกไฟล์ผ่าน dialog
  input.addEventListener('change', ()=>{
    if (input.files && input.files.length){ attachFiles(input.files); input.value = ''; }
  });

  // Drag & Drop
  ['dragenter','dragover'].forEach(evt =>
    dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add('is-dragover'); })
  );
  ['dragleave','drop'].forEach(evt =>
    dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('is-dragover'); })
  );
  dropzone.addEventListener('drop', e => attachFiles(e.dataTransfer.files));
  dropzone.addEventListener('keydown', (e)=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); input.click(); }});

  // Reset & Submit (เดโม่)
  resetBtn.addEventListener('click', ()=>{ resetFiles(); rerender(); });
  checkBtn.addEventListener('click', ()=> alert('Selected: ' + getFiles().map(f=>f.name).join(', ')));

  // เรนเดอร์ครั้งแรก
  rerender();
}