console.log('loaded: ui.js');
import { getFiles, hasFiles } from './state.js';
import { renderPdfPreview } from './pdf-preview.js';

// DOM อ้างอิง
const els = {
  grid: document.getElementById('gridSection'),
  dropCard: document.getElementById('dropCard'),
  checkBtn: document.getElementById('checkBtn'),
  fabStack: document.querySelector('.fab-stack'),
  resetBtn: document.getElementById('resetBtn'),
  pdfModal: document.getElementById('pdfModal'),
  pdfFrame: document.getElementById('pdfFrame'),
  modalClose: document.getElementById('modalClose'),
  modalBackdrop: document.getElementById('modalBackdrop'),
  modalTitle: document.getElementById('modalTitle'),
};

let previewURL = null;

export function toggleFabs(show){
  els.fabStack.classList.toggle('hidden', !show);
  els.resetBtn.classList.toggle('hidden', !show);
}

export function openPreview(file){
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  els.pdfFrame.src = previewURL;
  els.modalTitle.textContent = file.name;
  els.pdfModal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

export function closePreview(){
  els.pdfModal.classList.add('hidden');
  document.body.style.overflow = '';
  els.pdfFrame.src = '';
  if (previewURL) { URL.revokeObjectURL(previewURL); previewURL = null; }
}

export function bindModalClose(){
  els.modalClose.addEventListener('click', closePreview);
  els.modalBackdrop.addEventListener('click', closePreview);
  window.addEventListener('keydown', (e)=>{ if (e.key === 'Escape') closePreview(); });
}

export function render(onRemove, onCardClick){
  const files = getFiles();
  const has = hasFiles();

  els.dropCard.classList.toggle('hidden', has);
  els.grid.classList.toggle('hidden', !has);
  els.checkBtn.disabled = files.length < 1;
  toggleFabs(has);

  if(!has){ els.grid.innerHTML = ''; return; }

  els.grid.innerHTML = files.map(f => `
    <div class="pdf-card" data-id="${f.id}" title="Click to preview">
      <button class="remove" title="Remove">×</button>
      <div class="thumb" aria-hidden="true">
        <canvas></canvas>
        <div class="loading" aria-hidden="true"></div>
      </div>
      <div class="name">${f.name}</div>
    </div>
  `).join('');

  // ปุ่มลบ
  els.grid.querySelectorAll('.pdf-card .remove').forEach(btn=>{
    btn.addEventListener('click', (e)=>{
      const card = e.target.closest('.pdf-card');
      onRemove(card.getAttribute('data-id'));
      e.stopPropagation();
    });
  });

  // คลิกการ์ดเพื่อพรีวิว
  els.grid.querySelectorAll('.pdf-card').forEach(card=>{
    card.addEventListener('click', (e)=>{
      if (e.target.closest('.remove')) return;
      onCardClick(card.getAttribute('data-id'));
    });
  });

  // วาดภาพหน้าปก PDF
  els.grid.querySelectorAll('.pdf-card').forEach(card=>{
    const canvas = card.querySelector('canvas');
    const id = card.getAttribute('data-id');
    const file = files.find(x => x.id === id)?.file;
    if (file && canvas) renderPdfPreview(file, canvas);
  });
}