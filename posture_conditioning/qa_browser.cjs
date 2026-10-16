const {chromium}=require(process.argv[2]||'playwright');
const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');

(async()=>{
 const root=path.resolve(__dirname,'../runs/posture_v1/media');
 const qa=path.join(root,'qa');fs.mkdirSync(qa,{recursive:true});
 const manifest=JSON.parse(fs.readFileSync(path.join(root,'videos.json'),'utf8'));
 const browser=await chromium.launch({headless:true,channel:'msedge',args:['--allow-file-access-from-files']});
 const result={status:'passed',viewports:[],videos:[]};
 try{
  for(const width of [1440,390]){
   const page=await browser.newPage({viewport:{width,height:1000}});
   const errors=[];page.on('pageerror',e=>errors.push(e.message));
   await page.goto(pathToFileURL(path.join(root,'index.html')).href);
   await page.locator('img').evaluateAll(async imgs=>Promise.all(imgs.map(i=>i.decode())));
   const layout=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,
    corrupted:document.body.textContent.includes('\ufffd'),videos:document.querySelectorAll('video').length,
    textOverflow:[...document.querySelectorAll('h1,h2,p')].some(x=>x.scrollWidth>x.clientWidth+1),
    links:[...document.querySelectorAll('a')].map(a=>a.getAttribute('href'))}));
   if(layout.overflow||layout.corrupted||layout.textOverflow||layout.videos!==8)throw Error('Layout '+JSON.stringify(layout));
   for(const href of layout.links)if(!fs.existsSync(path.resolve(root,href)))throw Error('Missing link '+href);
   await page.screenshot({path:path.join(qa,'report_'+width+'.png')});
   const videos=await page.locator('video').all();
   for(let index=0;index<videos.length;index++){
    const v=videos[index];const expected=manifest.videos[index];
    await v.scrollIntoViewIfNeeded();
    const check=await v.evaluate(async(video,record)=>{
     const event=name=>new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(Error('Media event timeout: '+name)),15000);
      video.addEventListener(name,()=>{clearTimeout(timer);resolve();},{once:true});
     });
     video.muted=true;await video.play();
     const start=performance.now();
     while(video.currentTime<.1){if(performance.now()-start>10000)throw Error('Playback did not advance');await new Promise(r=>setTimeout(r,30));}
     video.pause();
     if(video.videoWidth!==1920||video.videoHeight!==880||Math.abs(video.duration-record.frames/20)>.1)throw Error('Invalid dimensions/duration');
     const canvas=document.createElement('canvas');canvas.width=1920;canvas.height=880;
     const context=canvas.getContext('2d',{willReadFrequently:true});
     const frames=[],counts=[],captures=[];
     for(const fraction of [.2,.5,.8]){
      const ready=event('seeked');video.currentTime=fraction*video.duration;await ready;
      context.drawImage(video,0,0);const data=context.getImageData(0,0,1920,880).data;
      const count=[0,0,0];
      for(let col=0;col<3;col++)for(let y=108;y<728;y++)for(let x=col*640+10;x<col*640+630;x++){
       const k=(y*1920+x)*4;if(data[k]-data[k+2]>14)count[col]++;
      }
      if(count.some(x=>x<500))throw Error('Blank mesh panel '+JSON.stringify(count));
      counts.push(count);frames.push(data);
      if(record.capture)captures.push(canvas.toDataURL('image/png').split(',')[1]);
     }
     const changed=[0,0,0];
     for(let col=0;col<3;col++)for(let y=108;y<728;y++)for(let x=col*640+10;x<col*640+630;x++){
      const k=(y*1920+x)*4;
      if(Math.abs(frames[0][k]-frames[2][k])+Math.abs(frames[0][k+1]-frames[2][k+1])+Math.abs(frames[0][k+2]-frames[2][k+2])>80)changed[col]++;
     }
     return {duration:video.duration,counts,changed,captures};
    },{...expected,capture:width===1440&&[0,3].includes(index)});
    for(let k=0;k<check.captures.length;k++)fs.writeFileSync(path.join(qa,`case_${index}_time_${k}.png`),Buffer.from(check.captures[k],'base64'));
    delete check.captures;
    if(check.changed.some(n=>n<100))throw Error('No visible motion in a panel '+index);
    result.videos.push({width,case:index,...check});
   }
   if(errors.length)throw Error(errors.join('\n'));
   result.viewports.push({width,...layout});await page.close();
  }
  const page=await browser.newPage({viewport:{width:1440,height:1200}});
  await page.goto(pathToFileURL(path.join(root,'index.html')).href);
  const posters=manifest.videos.map(v=>pathToFileURL(path.join(root,v.poster)).href);
  await page.setContent('<style>body{margin:0;display:grid;grid-template-columns:repeat(2,720px)}img{width:720px;display:block}</style>'+posters.map(src=>`<img src="${src}">`).join(''));
  await page.locator('img').evaluateAll(async imgs=>Promise.all(imgs.map(i=>i.decode())));
  await page.screenshot({path:path.join(qa,'contact_sheet.png'),fullPage:true});
  fs.writeFileSync(path.join(qa,'browser.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify({status:'passed',checks:result.videos.length}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
