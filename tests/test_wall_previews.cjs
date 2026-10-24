const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = readFileSync('index.html', 'utf8');
const scheduler = source.slice(source.indexOf('        let wallPreviewSession'), source.indexOf('        function getSlotCountForMode'));
async function check(size) {
    const images = Array.from({length:size}, (_,i)=>({dataset:{cameraId:String(i)},isConnected:true}));
    const timers = new Map();
    let sequence = 0, active = 0, peak = 0, revoked = 0;
    const visited = new Set();
    const context = vm.createContext({
        currentGridMode:'auto', API_URL:'http://test', AbortController,
        document:{hidden:false, querySelectorAll:()=>images,addEventListener(){}},
        window:{addEventListener(){}},
        URL: Object.assign(URL, {createObjectURL:()=> 'blob:' + sequence++, revokeObjectURL:()=>revoked++}),
        setTimeout(fn,ms){const id=sequence++;timers.set(id,{fn,ms});return id;},
        clearTimeout(id){timers.delete(id);},
        fetch:async (url,options)=>{
            active++;peak=Math.max(peak,active);
            assert.equal(options.cache,'no-store');
            visited.add(url.split('/').pop());
            await Promise.resolve();
            active--;
            return {ok:true,status:200,blob:async()=>({})};
        }
    });
    vm.runInContext(scheduler+';startWallPreviews(true)',context);
    const drain=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
    await drain();
    for(let i=0;i<Math.ceil(size/3)+2;i++){
        const timer=[...timers].find(([,t])=>t.ms===150);
        assert.ok(timer,'next preview batch scheduled');
        timers.delete(timer[0]);timer[1].fn();await drain();
    }
    assert.equal(visited.size,size,'all slots refreshed fairly');
    assert.ok(peak<=3,'at most three requests');
    vm.runInContext('stopWallPreviews()',context);
    assert.equal(timers.size,0,'no timers after stop');
    assert.ok(revoked>=size,'object URLs released');
    const count=visited.size;
    vm.runInContext('document.hidden=true;startWallPreviews(true)',context);
    await drain();
    assert.equal(visited.size,count);
    console.log(size+' slots: fairness, concurrency, cleanup and hidden tab passed');
}
(async()=>{for(const size of [4,9,16,36,182])await check(size);})().catch(e=>{console.error(e);process.exitCode=1;});

// Verify the streaming threshold, distribution, and fallback without camera hardware.
function checkStreaming() {
    for (const size of [9,16,17]) {
        const images=Array.from({length:size},(_,i)=>({
            dataset:{cameraId:String(i)},isConnected:true,
            removeAttribute(name){delete this[name];}
        }));
        let requests=0;
        const ctx=vm.createContext({
            currentGridMode:'auto', API_URL:'http://localhost:8081', AbortController, URL,
            document:{hidden:false,querySelectorAll:()=>images,addEventListener(){}},
            window:{addEventListener(){}},setTimeout:()=>1,clearTimeout(){},
            fetch:()=>{requests++;return new Promise(()=>{});}
        });
        vm.runInContext(scheduler+';startWallPreviews()',ctx);
        if(size<=16){
            const counts={};
            images.forEach(image=>{
                const host=new URL(image.src).hostname;
                counts[host]=(counts[host]||0)+1;
                assert.equal(image.crossOrigin,'anonymous');
            });
            assert.ok(Math.max(...Object.values(counts))<=4);
            assert.equal(requests,0);
            images[0].onerror();
            assert.equal(requests,3,'stream error switches to bounded fallback');
        } else assert.equal(requests,3,'large wall uses bounded requests');
        vm.runInContext('stopWallPreviews()',ctx);
        images.forEach(image=>assert.equal(image.src,undefined));
    }
    console.log('Streaming: 9/16 slots distributed; 17 slots and errors fall back; streams cleaned');
}
checkStreaming();
