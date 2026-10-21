const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require(path.resolve('.tools/posture-browser/node_modules/playwright'));

(async () => {
  const root=path.resolve('third_party/SALAD_1016/reproduction/artifacts/jitter_attribution_v3');
  const output=path.join(root,'qa');fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true,args:[
    '--allow-file-access-from-files','--disable-gpu','--renderer-process-limit=1','--js-flags=--max-old-space-size=128']});
  const records=[];const errors=[];
  try {
    for(const width of [1440,390]){
      const page=await browser.newPage({viewport:{width,height:1000}});
      page.on('pageerror',error=>errors.push(String(error)));
      await page.goto(pathToFileURL(path.join(root,'index.html')).href);
      const layout=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));
      if(layout.scrollWidth>width+1)throw Error('Document overflow '+JSON.stringify(layout));
      for(const element of await page.locator('img').all()){
        await element.scrollIntoViewIfNeeded();
        await element.evaluate(async img=>{await img.decode();if(img.naturalWidth<500)throw Error('Missing figure');});
      }
      const videos=[];
      for(const video of await page.locator('video').all()){
        await video.scrollIntoViewIfNeeded();
        const states=await video.evaluate(async v=>{
          if(v.readyState<1)await new Promise((resolve,reject)=>{v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',reject,{once:true});});
          const canvas=document.createElement('canvas');canvas.width=310;canvas.height=231;
          const context=canvas.getContext('2d',{willReadFrequently:true});
          const samples=[];
          for(const time of [0.05,v.duration/2,v.duration-.1]){
            v.currentTime=time;await new Promise(resolve=>v.addEventListener('seeked',resolve,{once:true}));
            context.drawImage(v,0,0,310,231);const data=context.getImageData(0,0,310,231).data;
            const panels=[];
            for(let row=0;row<2;row++)for(let col=0;col<3;col++){
              let body=0,sum=0;
              for(let y=row?122:16;y<(row?218:111);y++)for(let x=col*103+4;x<col*103+99;x++){
                const i=(y*310+x)*4;
                if(row?(data[i]-data[i+2]>12):(data[i]-data[i+2]>18))body++;
                sum+=data[i]+data[i+1]+data[i+2];
              }
              if(body<20)throw Error('Empty video panel '+row+' '+col+' '+body);
              panels.push({body,brightness:sum});
            }
            samples.push({time,panels,pixels:Array.from(data)});
          }
          let difference=0;for(let i=0;i<samples[0].pixels.length;i++)difference+=Math.abs(samples[0].pixels[i]-samples[1].pixels[i]);
          if(difference<1000)throw Error('Static video');
          v.currentTime=0;await new Promise(resolve=>v.addEventListener('seeked',resolve,{once:true}));
          await v.play();await new Promise(resolve=>setTimeout(resolve,250));
          if(v.paused || v.currentTime<=0)throw Error('Playback failed');v.pause();
          return {src:v.getAttribute('src'),duration:v.duration,width:v.videoWidth,height:v.videoHeight,
            samples:samples.map(({pixels,...rest})=>rest),motionDifference:difference,playback:true};
        });
        videos.push(states);
      }
      await page.evaluate(()=>scrollTo(0,0));
      await page.screenshot({path:path.join(output,'desktop_'+width+'.png')});
      records.push({viewport:width,layout,images:await page.locator('img').count(),videos});
      await page.close();
    }
    if(errors.length)throw Error(errors.join('\n'));
    fs.writeFileSync(path.join(output,'browser.json'),JSON.stringify({status:'passed',records,errors},null,2));
    console.log('BROWSER QA PASSED',records.length);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
