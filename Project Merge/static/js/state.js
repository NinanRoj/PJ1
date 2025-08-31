console.log('loaded: state.js');

// เก็บสถานะไฟล์แบบง่าย ๆ
let filesState = []; // {id, file, name}

export function uid(){ return Math.random().toString(36).slice(2,9); }
export function getFiles(){ return filesState.slice(); }
export function hasFiles(){ return filesState.length > 0; }

export function addFiles(list){
  const toAdd = Array.from(list).filter(f => f.type === 'application/pdf');
  toAdd.forEach(f => filesState.push({ id: uid(), file: f, name: f.name }));
}

export function removeFile(id){
  filesState = filesState.filter(x => x.id !== id);
}

export function resetFiles(){
  filesState = [];
}