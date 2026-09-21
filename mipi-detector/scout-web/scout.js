"use strict";
const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), focus = $('focus');
const ctx = overlay.getContext('2d'), focusCtx = focus.getContext('2d');
let config, state, source, timelineKey = '', focusedId = null, focusBox = null;
let lastFrame = 0, lastMessage = null, lastStateAt = 0;
let reconnectAfterPause = false;
const usbVideo=$('usb-video'), usbOverlay=$('usb-overlay'), usbCtx=usbOverlay.getContext('2d');
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
let usbSource, usbLastFrame=0, usbMessage=null, usbReconnect=false, selectedCamera='mipi';
let drawing=false, dragStart=null, draftZone=null;
const phaseNames = {idle:'STANDBY', searching:'FINDING SUBJECT', verifying:'INSPECTING', reviewed:'REVIEW READY'};
const number = value => typeof value === 'number' && Number.isFinite(value) ? value : null;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function reportError(text) { $('error').textContent = text || ''; $('error').hidden = !text; }
function metric(id, value, unit) {
  $(id).replaceChildren(document.createTextNode(value === null ? '—' : String(value)),
    Object.assign(document.createElement('small'),{textContent:' '+unit}));
}
function clearFocus() {
  focusCtx.clearRect(0,0,focus.width,focus.height);
  $('focus-empty').hidden = false;
  $('focus-id').textContent = 'NO SUBJECT';
  $('focus-label').textContent = 'Digital inspection view';
  $('focus-checks').textContent = 'Live detector view';
  focusedId = focusBox = null;
}
function drawFocus(obj) {
  if (!obj || video.readyState < 2) { clearFocus(); return; }
  const [x,y,w,h] = obj.bbox;
  const next = [Math.max(0,x-w*.14),Math.max(0,y-h*.12),Math.min(video.videoWidth-x+w*.14,w*1.28),Math.min(video.videoHeight-y+h*.12,h*1.24)];
  if (focusedId !== obj.id || !focusBox) focusBox = next;
  else focusBox = focusBox.map((n,i)=>n*.6+next[i]*.4);
  focusedId = obj.id;
  const [sx,sy,sw,sh] = focusBox;
  const scale = Math.min(focus.width/sw,focus.height/sh),dw=sw*scale,dh=sh*scale;
  focusCtx.fillStyle='#071018';focusCtx.fillRect(0,0,focus.width,focus.height);
  focusCtx.drawImage(video,sx,sy,sw,sh,(focus.width-dw)/2,(focus.height-dh)/2,dw,dh);
  $('focus-empty').hidden=true;$('focus-id').textContent=obj.id;
  $('focus-label').textContent=obj.label+' · '+Math.round(obj.confidence*100)+'% detection';
  $('focus-checks').textContent='Live candidate · YOLO';
}
function drawFrame(payload) {
  if (!payload.ready || !video.videoWidth) return;
  lastFrame=performance.now();$('video-empty').hidden=true;
  if (overlay.width!==video.videoWidth || overlay.height!==video.videoHeight) {
    overlay.width=video.videoWidth;overlay.height=video.videoHeight;
  }
  ctx.clearRect(0,0,overlay.width,overlay.height);
  const data=payload.message?.data;
  lastMessage=data || null;
  const current=data && state && data.mission_id===state.mission_id;
  const objects=data?.objects || [];
  const scale=overlay.width/1000;
  if (current && data.zone?.enabled) {
    const [x,y,w,h]=data.zone.bbox;
    ctx.fillStyle=data.zone.inside?'#ffc77928':'#70e4dc0e';ctx.strokeStyle=data.zone.inside?'#ffc779':'#70e4dc88';
    ctx.lineWidth=scale;ctx.setLineDash([8*scale,6*scale]);
    ctx.fillRect(x*overlay.width,y*overlay.height,w*overlay.width,h*overlay.height);
    ctx.strokeRect(x*overlay.width,y*overlay.height,w*overlay.width,h*overlay.height);ctx.setLineDash([]);
    ctx.font=`${10*scale}px sans-serif`;ctx.fillStyle=data.zone.inside?'#ffc779':'#70e4dc';
    ctx.fillText(data.zone.inside?'SUBJECT IN WATCH ZONE':'WATCH ZONE',x*overlay.width+8*scale,y*overlay.height+17*scale);
  }
  let focused=null;
  for (const obj of objects) {
    if (!Array.isArray(obj.bbox)||obj.bbox.length!==4) continue;
    const checking=current&&obj.id===state.candidate;
    const relevant=!state?.query||obj.label===state.object_class;
    const [x,y,w,h]=obj.bbox;
    const color=checking?'#ffc779':relevant?'#77ced8':'#8198a455';
    ctx.strokeStyle=color;ctx.lineWidth=(checking?2:1)*scale;ctx.strokeRect(x,y,w,h);
    if(relevant) {
      const label=`${obj.id} · ${obj.label}${checking?' · CHECKING':''}`;
      ctx.font=`${11*scale}px sans-serif`;
      const tw=ctx.measureText(label).width,ty=Math.max(0,y-22*scale);
      ctx.fillStyle='#07131be6';ctx.fillRect(x,ty,tw+14*scale,22*scale);
      ctx.fillStyle=color;ctx.fillText(label,x+7*scale,ty+15*scale);
    }
    if(relevant && obj.trail?.length>1) {
      ctx.strokeStyle='#9bea9988';ctx.lineWidth=1.4*scale;ctx.beginPath();
      obj.trail.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();
    }
    if(checking&&!focused) focused=obj;
  }
  // New capture sessions receive new IDs. Snapshot verdicts never label a live track.
  if(!focused && current && state.query) focused=objects.find(o=>o.id===focusedId&&o.label===state.object_class)
    || objects.filter(o=>o.label===state.object_class).sort((a,b)=>b.confidence-a.confidence)[0];
  drawFocus(focused);
  $('scene-count').textContent=`${objects.length} tracked object${objects.length===1?'':'s'}`;
}
function evidenceDialog(event) {
  $('evidence-image').src=event.image;$('evidence-title').textContent=event.title;
  $('evidence-detail').textContent=event.detail;
  $('evidence-time').textContent=`${event.camera_label||'MIPI 01'} · ${event.track_id || ''} · ${event.camera_id==='usb'?'Received':'Camera'} PTS ${event.source_pts_ms} ms · Captured ${new Date(event.captured_time*1000).toLocaleTimeString()}`;
  $('evidence-dialog').showModal();
}
function renderTimeline(events) {
  const key=events.map(e=>e.id).join(',');if(key===timelineKey)return;timelineKey=key;
  const root=$('timeline');root.replaceChildren();
  $('evidence-count').textContent=`${events.filter(e=>e.image).length} visual observations`;
  if(!events.length){
    const empty=document.createElement('div');empty.className='timeline-empty';
    const mark=document.createElement('span');mark.textContent='01 → 02 → 03';
    const detail=document.createElement('div'),title=document.createElement('strong'),text=document.createElement('p');
    title.textContent='The story of your mission, as it happens.';text.textContent='Saved images, visible evidence, and camera timestamps appear here.';
    detail.append(title,text);empty.append(mark,detail);root.append(empty);return;
  }
  for(const event of events){
    const card=document.createElement('article');card.className='event '+event.kind;
    if(event.image){const button=document.createElement('button'),image=document.createElement('img');image.src=event.image;image.alt='Saved evidence from '+(event.camera_label||'MIPI 01');button.title='Open saved camera evidence';button.append(image);button.onclick=()=>evidenceDialog(event);card.append(button);}
    const content=document.createElement('div'),time=document.createElement('small'),title=document.createElement('strong'),detail=document.createElement('p');
    time.textContent=(event.camera_label||'MIPI 01')+' · '+new Date(event.time*1000).toLocaleTimeString()+(event.track_id?' · '+event.track_id:'');
    title.textContent=event.title;detail.textContent=event.detail;content.append(time,title,detail);card.append(content);root.append(card);
  }
}
function renderState(next){
  const changed=!state||state.mission_id!==next.mission_id;
  if(changed&&next.query){$('query').value=next.query;$('object-class').value=next.object_class;$('zone').checked=next.zone.enabled;}
  state=next;lastStateAt=performance.now();
  const phase=state.phase;$('phase').textContent=phaseNames[phase]||phase;$('phase').className='phase '+phase;
  const live=state.camera.status==='live'&&state.camera.observation_age_s<2;
  const liveCount=Number(live)+Number(state.watch?.status==='live');
  $('connection').textContent=state.watch?`● ${liveCount}/2 CAMERAS LIVE`:live?'● SYSTEM LIVE':state.camera.status.toUpperCase();$('connection').className='connection'+(live?' live':'');
  const verdict=state.last_verdict;
  const messages={idle:'Describe a visible subject. Each check briefly pauses capture, then returns to live detection.',searching:`Waiting for a clear ${state.object_class}: ${state.query}`,verifying:`Gemma is checking a saved snapshot. Live capture resumes after the answer.`,reviewed:verdict?`${verdict.match==='yes'?'Snapshot matches':verdict.match==='no'?'Snapshot does not match':'A clearer view is needed'}. ${verdict.reason} Inspect again for a fresh observation.`:'Check inconclusive. Inspect another clear view.'};
  $('mission-message').textContent=state.vlm.status==='loading'?'Gemma is loading. The camera starts when the model is ready.':messages[phase]||'';
  $('scene-state').textContent=state.zone.inside?'WATCH ZONE · SUBJECT PRESENT':phase==='idle'?'LIVE OBSERVATION':phaseNames[phase];
  $('step-search').className=phase==='searching'?'active':phase!=='idle'?'done':'';
  $('step-verify').className=phase==='verifying'?'active':phase==='reviewed'?'done':'';
  $('step-follow').className=phase==='reviewed'?'active':'';
  const paused=['paused','resuming'].includes(state.camera.status);
  $('capture-pause').hidden=!paused;
  $('capture-pause').textContent=state.camera.status==='resuming'?'RETURNING TO LIVE CAPTURE':'SNAPSHOT CHECK · Live capture paused';
  if(paused){reconnectAfterPause=true;source?.expectSourcePause('Inspecting a saved snapshot',3500);ctx.clearRect(0,0,overlay.width,overlay.height);clearFocus();metric('metric-video',null,'fps');}
  if(live&&reconnectAfterPause){reconnectAfterPause=false;openVideo();}
  $('start').disabled=state.vlm.busy || state.vlm.status!=='ready';
  $('vlm-status').textContent=state.vlm.busy?'Examining evidence':({loading:'Loading model',ready:'Ready',error:'Unavailable'}[state.vlm.status]||state.vlm.status);
  $('vlm-dot').className=state.vlm.status==='ready'?'ready':'';$('retry').hidden=state.vlm.status!=='error';
  metric('metric-vlm',number(state.vlm.latency_s)?.toFixed(2)??null,'s');
  renderWatch(state.watch);
  renderPerformance();
  if(state.camera.error)reportError('Camera unavailable. See the application log.');else if(state.vlm.error)reportError('Gemma is unavailable. Retry the model or check the application log.');else reportError('');
  renderTimeline(state.events);
}
$('mission-form').addEventListener('submit',async event=>{
  event.preventDefault();$('start').disabled=true;
  try{renderState(await api('/api/mission',{query:$('query').value,object_class:$('object-class').value,zone:$('zone').checked}));clearFocus();}
  catch(error){reportError(error.message);}finally{$('start').disabled=false;}
});
$('reset').onclick=async()=>{try{renderState(await api('/api/reset',{}));clearFocus();}catch(e){reportError(e.message);}};
$('retry').onclick=async()=>{try{await api('/api/vlm/retry',{});$('retry').hidden=true;}catch(e){reportError(e.message);}};
document.querySelectorAll('[data-query]').forEach(button=>button.onclick=()=>{$('query').value=button.dataset.query;$('object-class').value=button.dataset.class;$('query').focus();});
$('close-dialog').onclick=()=>$('evidence-dialog').close();
$('evidence-dialog').onclick=event=>{if(event.target===$('evidence-dialog'))event.target.close();};
async function poll(){try{renderState(await api('/api/state'));}catch(e){reportError('Connection to Modalix lost. Reconnecting…');$('connection').textContent='DISCONNECTED';$('connection').className='connection';}setTimeout(poll,350);}
function openVideo(){
  // Re-negotiate after a declared pause to clear the browser's old frame buffer.
  source?.stop();lastFrame=0;lastMessage=null;clearFocus();
  ctx.clearRect(0,0,overlay.width,overlay.height);$('video-empty').hidden=false;
  source=InsightSource.open({channel:config.channel,video,syncBufferMs:180,retentionMs:1200,holdMs:0,onFrame:drawFrame,
    onStatus:status=>{$('video-status').textContent=status.width?`${status.width} × ${status.height} · ${status.text}`:status.text;if(selectedCamera==='mipi')metric('metric-video',status.phase==='live'?number(status.fps):null,'fps');}});
}

function selectCamera(id){
  if(id==='usb'&&!config?.usb)return;
  selectedCamera=id;
  $('primary-camera').append($(id==='mipi'?'mipi-panel':'usb-panel'));
  if(config?.usb)$('secondary-camera').append($(id==='mipi'?'usb-panel':'mipi-panel'));
  $('find-panel').hidden=id!=='mipi';$('watch-panel').hidden=id!=='usb';
  document.querySelector('.focus-panel').hidden=id!=='mipi';
  for(const camera of ['mipi','usb']){
    $('select-'+camera).classList.toggle('selected',camera===id);
    $('select-'+camera).setAttribute('aria-pressed',String(camera===id));
  }
  renderPerformance();
}
function renderPerformance(){
  if(!state)return;
  const camera=selectedCamera==='usb'?state.watch:state.camera;
  const live=camera?.status==='live'&&camera.observation_age_s<2;
  metric('metric-detection',live?camera.fps:null,'fps');
  metric('metric-age',live?Math.round(camera.observation_age_s*1000):null,'ms');
  const status=(selectedCamera==='usb'?usbSource:source)?.status();
  metric('metric-video',live&&status?.phase==='live'?number(status.fps):null,'fps');
}
function renderWatch(watch){
  if(!watch)return;
  $('mipi-strip-status').textContent=state.camera.status==='live'?`${state.camera.fps} fps · live`:state.camera.status;
  $('usb-strip-status').textContent=watch.status==='live'?`${watch.fps} fps · ${watch.occupied?'occupied':'observing'}`:watch.status;
  $('watch-enabled').checked=watch.enabled;$('watch-class').value=watch.object_class;
  const label=watch.status!=='live'?watch.status.toUpperCase():!watch.enabled?'WATCH OFF':watch.occupied===null?'OBSERVING':watch.occupied?'OCCUPIED':'NO WATCHED OBJECTS';
  $('watch-status').textContent=label;
  $('watch-status').className='phase '+(watch.occupied?'occupied':watch.occupied===false?'clear':'');
  $('usb-scene-state').textContent=label;
  $('usb-rate').textContent=watch.status==='live'?`${watch.fps} detection fps`:'No live observations';
  $('inspect-area').disabled=state.vlm.busy||state.vlm.status!=='ready'||watch.status!=='live'||watch.inspection_requested;
  $('watch-vlm-status').textContent=watch.checking?'Inspecting area':state.vlm.busy?'Checking MIPI snapshot':state.vlm.status;
  $('watch-message').textContent=watch.error?'USB camera unavailable. Reconnect it; the MIPI view can continue.':
    watch.checking?'Gemma is examining the saved area crop. Both cameras resume after the answer.':
    watch.last_verdict?`Last snapshot: ${watch.last_verdict.reason} Live occupancy comes from current detections.`:
    watch.occupied?'An object overlaps the marked area. Its image is saved in the mission timeline.':
    'Events report detected objects in this area. Unrecognized objects may not be detected.';
  const paused=['paused','resuming'].includes(watch.status);
  $('usb-pause').hidden=!paused;
  if(paused){usbReconnect=true;usbSource?.expectSourcePause('Inspecting a saved snapshot',6000);usbCtx.clearRect(0,0,usbOverlay.width,usbOverlay.height);}
  if(watch.status==='live'&&usbReconnect){usbReconnect=false;openUsbVideo();}
}
function drawUsb(payload){
  if(!payload.ready||!usbVideo.videoWidth)return;
  usbLastFrame=performance.now();$('usb-empty').hidden=true;
  if(usbOverlay.width!==usbVideo.videoWidth||usbOverlay.height!==usbVideo.videoHeight){usbOverlay.width=usbVideo.videoWidth;usbOverlay.height=usbVideo.videoHeight;}
  usbCtx.clearRect(0,0,usbOverlay.width,usbOverlay.height);
  usbMessage=payload.message?.data||null;
  const current=usbMessage&&state?.watch&&usbMessage.generation===state.watch.generation;
  const objects=current?usbMessage.objects||[]:[];
  const scale=usbOverlay.width/800;
  for(const obj of objects){
    const [x,y,w,h]=obj.bbox;
    usbCtx.strokeStyle=obj.inside?'#ffc779':'#77ced8';usbCtx.lineWidth=scale;
    usbCtx.strokeRect(x,y,w,h);usbCtx.font=`${11*scale}px sans-serif`;
    usbCtx.fillStyle='#07131be6';usbCtx.fillRect(x,Math.max(0,y-21*scale),150*scale,21*scale);
    usbCtx.fillStyle=obj.inside?'#ffc779':'#77ced8';
    usbCtx.fillText(`${obj.id} · ${obj.label}`,x+5*scale,Math.max(15*scale,y-6*scale));
  }
  const zone=draftZone||state?.watch?.zone;
  if(zone&&(state?.watch?.enabled||drawing)){
    const [x,y,w,h]=zone.map((v,i)=>v*(i%2?usbOverlay.height:usbOverlay.width));
    const occupied=current&&state.watch.status==='live'&&state.watch.enabled&&usbMessage.occupied===true;
    // One dark-red pulse per second, independent of camera frame rate.
    const warningAlpha=reducedMotion.matches ? .42 : .3+.18*Math.cos(performance.now()*2*Math.PI/1000);
    usbCtx.fillStyle=occupied?`rgba(139, 12, 28, ${warningAlpha})`:'#70e4dc10';
    usbCtx.strokeStyle=occupied?'#ff626e':'#70e4dc';
    usbCtx.lineWidth=2*scale;usbCtx.setLineDash([7*scale,5*scale]);
    usbCtx.fillRect(x,y,w,h);usbCtx.strokeRect(x,y,w,h);usbCtx.setLineDash([]);
  }
  $('usb-count').textContent=`${objects.filter(o=>o.inside).length} objects in area`;
}
function openUsbVideo(){
  if(!config?.usb)return;
  usbSource?.stop();usbLastFrame=0;usbMessage=null;$('usb-empty').hidden=false;
  usbSource=InsightSource.open({channel:config.usb.channel,video:usbVideo,syncBufferMs:180,retentionMs:1200,holdMs:0,onFrame:drawUsb,
    onStatus:status=>{$('usb-video-status').textContent=status.width?`${status.width} × ${status.height} · ${status.text}`:status.text;if(selectedCamera==='usb')metric('metric-video',status.phase==='live'?number(status.fps):null,'fps');}});
}
async function updateWatch(fields){
  try{renderState(await api('/api/watch',fields));}catch(e){reportError(e.message);}
}
$('select-mipi').onclick=()=>selectCamera('mipi');$('select-usb').onclick=()=>selectCamera('usb');
$('watch-enabled').onchange=()=>updateWatch({enabled:$('watch-enabled').checked});
$('watch-class').onchange=()=>updateWatch({object_class:$('watch-class').value});
$('inspect-area').onclick=async()=>{try{renderState(await api('/api/watch/inspect',{}));}catch(e){reportError(e.message);}};
$('draw-area').onclick=()=>{drawing=!drawing;draftZone=dragStart=null;usbOverlay.classList.toggle('drawing',drawing);$('draw-area').classList.toggle('active',drawing);$('area-hint').textContent=drawing?'Drag across the USB image to set the watch area.':'Drag a rectangle on the USB view to match your mat or table.';};
function usbPoint(event){const r=usbOverlay.getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))];}
usbOverlay.onpointerdown=event=>{if(!drawing)return;dragStart=usbPoint(event);usbOverlay.setPointerCapture(event.pointerId);};
usbOverlay.onpointermove=event=>{if(!dragStart)return;const p=usbPoint(event);draftZone=[Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.abs(p[0]-dragStart[0]),Math.abs(p[1]-dragStart[1])];};
usbOverlay.onpointerup=async event=>{if(!dragStart)return;const zone=draftZone;dragStart=null;usbOverlay.releasePointerCapture(event.pointerId);if(zone&&zone[2]>=.05&&zone[3]>=.05){await updateWatch({zone});drawing=false;usbOverlay.classList.remove('drawing');$('draw-area').classList.remove('active');$('area-hint').textContent='Watch area updated. Draw again to adjust it.';}else{$('area-hint').textContent='Draw a larger area: at least 5% of the image in each direction.';}draftZone=null;};
usbOverlay.onpointercancel=()=>{dragStart=draftZone=null;};
async function start(){
  try{
    config=await api('/api/config');
    $('camera-strip').hidden=!config.usb;$('secondary-camera').hidden=!config.usb;
    openVideo();
    openUsbVideo();
    window.scout={get state(){return state;},get lastMessage(){return lastMessage;},get source(){return source;},get usbSource(){return usbSource;},get usbMessage(){return usbMessage;}};
    poll();
  }catch(error){reportError(error.message);setTimeout(start,2500);}
}
setInterval(()=>{
  if(lastFrame&&performance.now()-lastFrame>1500){ctx.clearRect(0,0,overlay.width,overlay.height);clearFocus();$('scene-state').textContent='WAITING FOR LIVE VIDEO';metric('metric-video',null,'fps');}
  if(lastStateAt&&performance.now()-lastStateAt>2500){$('connection').textContent='DISCONNECTED';$('connection').className='connection';}
  if(usbLastFrame&&performance.now()-usbLastFrame>1500){usbCtx.clearRect(0,0,usbOverlay.width,usbOverlay.height);$('usb-scene-state').textContent='WAITING FOR LIVE VIDEO';$('usb-count').textContent='No current observation';}
},500);
window.addEventListener('beforeunload',()=>{source?.stop();usbSource?.stop();});
start();
