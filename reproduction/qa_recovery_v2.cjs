const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||path.resolve(__dirname,'../../../.tools/posture-browser/node_modules/playwright'));

(async()=>{
 const root=path.resolve(__dirname,'artifacts/physics_v1_20261016/recovery_v2');
 const qa=path.join(root,'qa');fs.mkdirSync(qa,{recursive:true});
 const manifest=JSON.parse(fs.readFileSync(path.join(root,'videos.json'),'utf8'));
 const official=JSON.parse(fs.readFileSync(path.join(root,'official_videos.json'),'utf8'));
 const browser=await chromium.launch({headless:true,channel:'msedge',args:['--allow-file-access-from-files']});
 const result={status:'passed',viewports:[],video_checks:[],naturalness_verified:false};
 async function inspect(video,sixPanels,capture){
  return video.evaluate(async(v,options)=>{
   const wait=event=>new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Video timeout '+event)),15000);
    v.addEventListener(event,()=>{clearTimeout(timer);resolve();},{once:true});
    v.addEventListener('error',()=>{clearTimeout(timer);reject(Error('Video decode error'));},{once:true});
   });
   v.muted=true;await v.play();const start=performance.now();
   while(v.currentTime<.1){
    if(performance.now()-start>10000)throw Error('Video did not advance '+JSON.stringify({src:v.src,time:v.currentTime,ready:v.readyState,network:v.networkState,paused:v.paused,error:v.error?.message}));
    await new Promise(r=>setTimeout(r,30));
   }
   v.pause();
   if(Math.abs(v.duration-6)>.1)throw Error('Unexpected duration');
   if(options.sixPanels&&(v.videoWidth!==1920||v.videoHeight!==1460))throw Error('Unexpected diagnostic dimensions');
   const canvas=document.createElement('canvas');canvas.width=v.videoWidth;canvas.height=v.videoHeight;
   const ctx=canvas.getContext('2d',{willReadFrequently:true});
   const frames=[],counts=[],captures=[];
   const regions=options.sixPanels?Array.from({length:6},(_,i)=>({x:(i%3)*640+10,y:100+Math.floor(i/3)*700,w:620,h:620})):
    [{x:0,y:0,w:v.videoWidth,h:v.videoHeight}];
   for(const fraction of [.2,.5,.8]){
    const done=wait('seeked');v.currentTime=fraction*v.duration;await done;ctx.drawImage(v,0,0);
    const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
    const colored=regions.map(r=>{
     let n=0;for(let y=r.y;y<r.y+r.h;y++)for(let x=r.x;x<r.x+r.w;x++){
      const i=(y*canvas.width+x)*4;
      if(Math.max(pixels[i],pixels[i+1],pixels[i+2])-Math.min(pixels[i],pixels[i+1],pixels[i+2])>35)n++;
     }return n;
    });
    if(colored.some(n=>n<400))throw Error('Blank body or skeleton '+JSON.stringify(colored));
    counts.push(colored);frames.push(pixels);
    if(options.capture)captures.push(canvas.toDataURL('image/png').split(',')[1]);
   }
   const changes=regions.map(r=>{
    let n=0;for(let y=r.y;y<r.y+r.h;y++)for(let x=r.x;x<r.x+r.w;x++){
     const i=(y*canvas.width+x)*4;
     if(Math.abs(frames[0][i]-frames[2][i])+Math.abs(frames[0][i+1]-frames[2][i+1])+Math.abs(frames[0][i+2]-frames[2][i+2])>100)n++;
    }return n;
   });
   if(changes.some(n=>n<300))throw Error('Motion absent '+JSON.stringify(changes));
   return {width:v.videoWidth,height:v.videoHeight,duration:v.duration,colored:counts,changed_pixels:changes,captures};
  },{sixPanels,capture});
 }
 try{
  for(const width of [1440,390]){
   const page=await browser.newPage({viewport:{width,height:1050}});const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(pathToFileURL(path.join(root,'index.html')).href);
   const layout=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,
    corrupted:document.body.textContent.includes('\ufffd'),videos:document.querySelectorAll('video').length,
    textOverflow:[...document.querySelectorAll('h1,h2,p')].some(x=>x.scrollWidth>x.clientWidth+1),
    links:[...document.querySelectorAll('a')].map(a=>a.getAttribute('href'))}));
   if(layout.overflow||layout.corrupted||layout.textOverflow||layout.videos!==4)throw Error('Layout failed '+JSON.stringify(layout));
   for(const href of layout.links)if(!fs.existsSync(path.resolve(root,href)))throw Error('Missing link '+href);
   await page.screenshot({path:path.join(qa,'report_'+width+'.png')});
   for(const row of manifest.videos){
    const card=page.locator(`article[data-motion="${row.name}"]`);
    const video=card.locator('video');await video.scrollIntoViewIfNeeded();
    const check=await inspect(video,true,width===1440);
    for(let i=0;i<check.captures.length;i++)fs.writeFileSync(path.join(qa,`${row.name}_time${i}.png`),Buffer.from(check.captures[i],'base64'));
    delete check.captures;result.video_checks.push({viewport:width,name:row.name,kind:'six_panels',...check});
    console.log(JSON.stringify({viewport:width,name:row.name,kind:'six_panels',status:'passed'}));
   }
   for(const row of official.videos){
    const element=await page.evaluateHandle(src=>{
     const v=document.createElement('video');v.src=src;v.controls=true;v.style.width='100%';v.style.maxWidth='1000px';
     document.querySelector('main').append(v);return v;
    },row.path);
    const check=await inspect(element.asElement(),false,width===1440);
    for(let i=0;i<check.captures.length;i++)fs.writeFileSync(path.join(qa,`${row.name}_official_time${i}.png`),Buffer.from(check.captures[i],'base64'));
    delete check.captures;result.video_checks.push({viewport:width,name:row.name,kind:'official_skeleton',...check});
    console.log(JSON.stringify({viewport:width,name:row.name,kind:'official_skeleton',status:'passed'}));
    await element.evaluate(v=>v.remove());
   }
   if(errors.length)throw Error(errors.join('\n'));
   result.viewports.push({width,...layout});await page.close();
  }
  fs.writeFileSync(path.join(qa,'browser.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify({status:'passed',video_checks:result.video_checks.length,viewports:[1440,390],naturalness_verified:false}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
