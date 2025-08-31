console.log('loaded: pdf-preview.js');

// ใช้ pdf.js จาก CDN (ตัวแปร pdfjsLib เป็น global)
export async function renderPdfPreview(file, canvas){
  try{
    const buf = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(buf) });
    const pdf = await loadingTask.promise;
    const page = await pdf.getPage(1);

    const ratio = window.devicePixelRatio || 1;
    const cont = canvas.parentElement;
    const maxW = cont.clientWidth;
    const maxH = cont.clientHeight;

    const vp0 = page.getViewport({ scale: 1 });
    const scale = Math.min((maxW*ratio)/vp0.width, (maxH*ratio)/vp0.height);
    const vp = page.getViewport({ scale });

    canvas.width = Math.floor(vp.width);
    canvas.height = Math.floor(vp.height);
    canvas.style.width = (vp.width/ratio) + 'px';
    canvas.style.height = (vp.height/ratio) + 'px';

    const ctx = canvas.getContext('2d', { alpha: false });
    await page.render({ canvasContext: ctx, viewport: vp }).promise;
    await pdf.destroy();
  }catch(err){
    console.error('PDF preview error:', err);
  }finally{
    const loader = canvas.parentElement.querySelector('.loading');
    if (loader) loader.remove();
  }
}