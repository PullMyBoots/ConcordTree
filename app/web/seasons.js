'use strict';
// A fixed landscape: weather, foliage and light share one continuous seasonal phase.
window.seasonLandscape = canvas => {
  const ctx=canvas.getContext('2d'),cycle=600000,tau=Math.PI*2;
  const names=['spring','summer','autumn','winter'];
  const palettes=[
    {sky:'#254746',haze:'#82977d',far:'#426052',hill:'#304c3b',land:'#203a29',water:'#355a50',bark:'#b2ad8b',leaf:'#9cbd79',flower:'#f0d5d4',snow:'#d6e2df',card:'#20382c',accent:'#dfdfbc',line:'#8ea580'},
    {sky:'#172c3b',haze:'#5d7478',far:'#344d50',hill:'#26483c',land:'#17372b',water:'#294951',bark:'#8d9e86',leaf:'#679478',flower:'#f0d9df',snow:'#c7dcdb',card:'#1c3433',accent:'#c1d9ce',line:'#7c9991'},
    {sky:'#353e49',haze:'#aa9274',far:'#655f52',hill:'#524d3d',land:'#35392b',water:'#48514f',bark:'#b8a188',leaf:'#d29a57',flower:'#dab077',snow:'#d1d4d6',card:'#352f28',accent:'#ead3ae',line:'#b79c7b'},
    {sky:'#263b50',haze:'#899fac',far:'#4c6574',hill:'#3c5664',land:'#293f49',water:'#3b5668',bark:'#a5b4be',leaf:'#9eabb7',flower:'#bcd2dd',snow:'#e1edf1',card:'#263440',accent:'#d2e1e9',line:'#8babbc'}
  ];
  const colors=palettes.map(p=>Object.fromEntries(Object.entries(p).map(([k,v])=>[k,[1,3,5].map(i=>parseInt(v.slice(i,i+2),16))])));
  const mix=(a,b,t)=>a+(b-a)*t,clamp=v=>Math.max(0,Math.min(1,v)),smooth=t=>t*t*(3-2*t);
  const rgb=(v,a=1)=>`rgba(${v.map(Math.round).join(',')},${a})`;
  let width=0,height=0,trees=[],clouds=[],lastStyle=-Infinity;
  function random(seed){return()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};}
  function makeCloud(dark) {
    const image=document.createElement('canvas');image.width=640;image.height=240;
    const c=image.getContext('2d'),rand=random(47);
    // Soft volumes are cached once; no per-frame blur or external assets.
    for(let i=0;i<14;i++) {
      const x=100+rand()*440,y=85+rand()*50,r=35+rand()*55;
      c.save();c.translate(x,y);c.scale(1,.67);
      const gradient=c.createRadialGradient(0,-r*.15,r*.12,0,0,r);
      gradient.addColorStop(0,dark?'#71838b85':'#dae5d4a6');
      gradient.addColorStop(.55,dark?'#354a5bba':'#b2c2b3a1');
      gradient.addColorStop(1,dark?'#283d4c00':'#adbfad00');
      c.fillStyle=gradient;c.beginPath();c.arc(0,0,r,0,tau);c.fill();c.restore();
    }
    return image;
  }
  const fairCloud=makeCloud(false),stormCloud=makeCloud(true);
  function makeTree(x,y,size,seed,lean=0) {
    const rand=random(seed),branches=[],leaves=[],fruits=[];
    function branch(px,py,length,angle,depth,thickness) {
      const ex=px+Math.cos(angle)*length,ey=py+Math.sin(angle)*length,curve=(rand()-.5)*length*.32;
      branches.push({px,py,ex,ey,cx:(px+ex)/2+curve,cy:(py+ey)/2,thickness,depth});
      if(depth<6) {
        const delta=.30+rand()*.24;
        branch(ex,ey,length*(.67+rand()*.1),angle-delta,depth+1,thickness*.66);
        branch(ex,ey,length*(.65+rand()*.12),angle+delta*.85,depth+1,thickness*.64);
      } else {
        if(rand()>.42)fruits.push({x:ex,y:ey+size*.012,r:size*(.006+rand()*.003),phase:rand()*tau});
        for(let i=0;i<9;i++)leaves.push({x:ex+(rand()-.5)*size*.15,y:ey+(rand()-.5)*size*.12,r:size*(.009+rand()*.009),angle:rand()*tau,phase:rand()*tau,threshold:rand()});
      }
    }
    branch(x,y,size*.30,-Math.PI/2+lean,0,size*.023);
    return {x,y,size,seed,branches,leaves,fruits};
  }
  function resize() {
    width=innerWidth;height=innerHeight;const dpr=Math.min(devicePixelRatio||1,2);
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx?.setTransform(dpr,0,0,dpr,0,0);
    const size=Math.min(height*.72,width*.40);
    trees=[makeTree(width*.12,height*.87,size,81,-.06),makeTree(width*.89,height*.82,size*.84,249,.09)];
    clouds=Array.from({length:8},(_,i)=>({x:((i*.372+.03)%1)*width,y:height*(.06+(i%3)*.07),w:width*(.28+(i%3)*.07),phase:i*1.91}));
  }
  function paint(milliseconds,phaseOverride=null,force=false) {
    if(!ctx||!width)return;
    const raw=phaseOverride===null?milliseconds/cycle*4:phaseOverride,phase=((raw%4)+4)%4;
    const index=Math.floor(phase),t=smooth(phase-index),next=(index+1)%4;
    const p=Object.fromEntries(Object.keys(colors[0]).map(k=>[k,colors[index][k].map((v,i)=>mix(v,colors[next][k][i],t))]));
    const weights=[0,0,0,0];weights[index]=1-t;weights[next]+=t;
    const [spring,summer,autumn,winter]=weights,time=milliseconds/1000;
    const wind=.20*spring+.50*summer+autumn+.32*winter;
    if(force || Math.abs(milliseconds-lastStyle)>150 || milliseconds===0) {
      for(const key of ['card','accent','line','sky'])document.body.style.setProperty('--season-'+key,rgb(p[key]));
      document.body.dataset.season=names[index];lastStyle=milliseconds;
    }
    ctx.globalAlpha=1;ctx.lineCap='round';ctx.lineJoin='round';
    const sky=ctx.createLinearGradient(0,0,0,height);sky.addColorStop(0,rgb(p.sky));sky.addColorStop(.66,rgb(p.haze));sky.addColorStop(1,rgb(p.land));ctx.fillStyle=sky;ctx.fillRect(0,0,width,height);
    // Spring sunlight retreats behind gathering summer clouds.
    const sunX=width*.79,sunY=height*.18,sunR=Math.min(width,height)*.034,sunlight=.75*spring+.11*autumn+.07*winter;
    if(sunlight>.001) {
      const glow=ctx.createRadialGradient(sunX,sunY,sunR*.3,sunX,sunY,sunR*9);glow.addColorStop(0,`rgba(250,227,165,${sunlight*.34})`);glow.addColorStop(1,'#f5d69b00');ctx.fillStyle=glow;ctx.fillRect(0,0,width,height*.75);
      ctx.fillStyle=`rgba(247,231,183,${sunlight})`;ctx.beginPath();ctx.arc(sunX,sunY,sunR,0,tau);ctx.fill();
      for(let i=0;i<7;i++) {
        const a=1.2+i*.24,ray=ctx.createLinearGradient(sunX,sunY,sunX+Math.cos(a)*height*.65,sunY+Math.sin(a)*height*.65);
        ray.addColorStop(0,`rgba(247,231,180,${spring*.045})`);ray.addColorStop(1,'#f5e7b400');ctx.strokeStyle=ray;ctx.lineWidth=12+i*4;
        ctx.beginPath();ctx.moveTo(sunX,sunY);ctx.lineTo(sunX+Math.cos(a)*height*.65,sunY+Math.sin(a)*height*.65);ctx.stroke();
      }
    }
    const cloudCover=.13*spring+.95*summer+.78*autumn+.58*winter;
    for(const cloud of clouds) {
      const x=cloud.x-cloud.w/2+Math.sin(time*.025+cloud.phase)*(12+wind*24),y=cloud.y+Math.sin(time*.017+cloud.phase)*5,h=cloud.w*.35;
      ctx.globalAlpha=cloudCover*(.58+.30*summer);ctx.drawImage(fairCloud,x,y,cloud.w,h);
      ctx.globalAlpha=summer*.85+autumn*.15+winter*.10;ctx.drawImage(stormCloud,x-8,y+4,cloud.w*1.15,h*1.2);
    }
    ctx.globalAlpha=1;
    // Slow, distant lightning; no whole-screen flashing and no audio.
    if(summer>.002) {
      const beat=(time+9)%23,flash=smooth(clamp(1-Math.abs(beat-10)/1.1));
      ctx.strokeStyle=`rgba(218,231,239,${summer*(.08+.43*flash)})`;ctx.lineWidth=1.3;
      ctx.beginPath();ctx.moveTo(width*.86,height*.21);ctx.lineTo(width*.835,height*.28);ctx.lineTo(width*.854,height*.275);ctx.lineTo(width*.818,height*.38);ctx.stroke();
      ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(width*.835,height*.28);ctx.lineTo(width*.80,height*.30);ctx.lineTo(width*.786,height*.35);ctx.stroke();
    }
    function shape(color,points,curved=false) {
      ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(0,height);ctx.lineTo(points[0][0]*width,points[0][1]*height);
      for(let i=1;i<points.length;i++) {
        const a=points[i-1],b=points[i];
        if(curved)ctx.bezierCurveTo(width*mix(a[0],b[0],.45),height*a[1],width*mix(a[0],b[0],.55),height*b[1],width*b[0],height*b[1]);
        else ctx.lineTo(width*b[0],height*b[1]);
      }
      ctx.lineTo(width,height);ctx.closePath();ctx.fill();
    }
    shape(rgb(p.far),[[0,.53],[.09,.46],[.16,.49],[.29,.38],[.39,.46],[.48,.43],[.64,.52],[.79,.40],[.88,.48],[1,.44]]);
    if(winter>.002) {
      ctx.fillStyle=rgb(p.snow,winter*.38);
      for(const [x,y] of [[.29,.38],[.79,.40],[.48,.43]]){ctx.beginPath();ctx.moveTo(width*x,height*y);ctx.lineTo(width*(x-.034),height*(y+.048));ctx.lineTo(width*(x-.008),height*(y+.032));ctx.lineTo(width*(x+.014),height*(y+.049));ctx.lineTo(width*(x+.039),height*(y+.033));ctx.closePath();ctx.fill();}
    }
    shape(rgb(p.hill),[[0,.65],[.11,.59],[.20,.62],[.32,.54],[.46,.59],[.61,.57],[.76,.62],[.90,.54],[1,.57]],true);
    // The far bank keeps its miniature grove through every season.
    for(let i=0;i<53;i++) {
      const x=width*i/52,y=height*(.65+Math.sin(i*.61)*.012),h=height*(.013+(i%5)*.004);
      ctx.fillStyle=rgb(p.land,.47);ctx.beginPath();ctx.moveTo(x,y-h);ctx.lineTo(x-h*.38,y);ctx.lineTo(x+h*.38,y);ctx.closePath();ctx.fill();
      if(winter>.002){ctx.strokeStyle=rgb(p.snow,winter*.28);ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(x-h*.28,y-h*.22);ctx.lineTo(x,y-h);ctx.stroke();}
    }
    ctx.fillStyle=rgb(p.water);ctx.fillRect(0,height*.69,width,height*.31);
    for(let i=0;i<30;i++) {
      const y=height*(.70+i*.009),x=width*(.5+Math.sin(i*11)*.46),span=width*(.025+(i%5)*.013);
      ctx.strokeStyle=rgb(p.accent,.03+(i%3)*.014);ctx.lineWidth=.6;
      ctx.beginPath();ctx.moveTo(x-span+Math.sin(time*(.25+wind*.20)+i)*(3+wind*5),y);ctx.lineTo(x+span,y);ctx.stroke();
    }
    const bank=[[0,.81],[.09,.80],[.19,.86],[.31,.92],[.47,.99],[.62,.97],[.76,.89],[.90,.79],[1,.77]];
    shape(rgb(p.land),bank,true);
    if(winter>.002)shape(rgb(p.snow,winter*.55),bank,true);
    // Grasses and fallen leaves slowly disappear beneath winter snow.
    for(let i=0;i<90;i++) {
      const x=width*(i/89),y=height*(.94+.04*Math.sin(i*17)),h=height*(.008+(i%5)*.003),sway=Math.sin(time*.6+i)*wind*5;
      ctx.strokeStyle=rgb(p.leaf,(1-winter)*.28);ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(x,y);ctx.quadraticCurveTo(x+sway,y-h*.7,x-2+sway,y-h);ctx.stroke();
      if(i%2===0&&autumn>.002){ctx.fillStyle=`rgba(196,131,66,${autumn*.4})`;ctx.beginPath();ctx.ellipse(x,y,3.2,1.2,i,0,tau);ctx.fill();}
    }
    ctx.lineCap='round';
    function bend(tree,x,y) {
      return x+Math.sin(time*.55+tree.seed)*wind*9*Math.pow(clamp((tree.y-y)/tree.size),1.4);
    }
    for(const tree of trees) {
      for(const b of tree.branches) {
        const px=bend(tree,b.px,b.py),cx=bend(tree,b.cx,b.cy),ex=bend(tree,b.ex,b.ey);
        ctx.strokeStyle=rgb(p.bark,b.depth ? .53 : .73);ctx.lineWidth=b.thickness;
        ctx.beginPath();ctx.moveTo(px,b.py);ctx.quadraticCurveTo(cx,b.cy,ex,b.ey);ctx.stroke();
        if(winter>.002&&b.depth>1) {
          ctx.strokeStyle=rgb(p.snow,winter*.83);ctx.lineWidth=Math.max(.65,b.thickness*.65);
          ctx.beginPath();ctx.moveTo(px,b.py-b.thickness*.4);ctx.quadraticCurveTo(cx,b.cy-b.thickness*.4,ex,b.ey-b.thickness*.3);ctx.stroke();
        }
      }
      tree.leaves.forEach((leaf,i)=>{
        const foliage=clamp(spring*.85+summer+autumn*(.12+leaf.threshold*.55));
        if(foliage<.002)return;
        const growth=.25+.75*Math.sqrt(foliage),sway=Math.sin(time*(.45+wind*.3)+leaf.phase)*(1+wind*3);
        ctx.save();ctx.translate(bend(tree,leaf.x,leaf.y)+sway,leaf.y);ctx.rotate(leaf.angle+Math.sin(time*.4+leaf.phase)*(.06+wind*.12));
        ctx.fillStyle=rgb(p.leaf,foliage*(.42+leaf.threshold*.34));ctx.beginPath();ctx.ellipse(0,0,leaf.r*1.4*growth,leaf.r*.68*growth,0,0,tau);ctx.fill();
        if(i%7===0) {ctx.strokeStyle=rgb(p.bark,foliage*.21);ctx.lineWidth=.45;ctx.beginPath();ctx.moveTo(-leaf.r,0);ctx.lineTo(leaf.r,0);ctx.stroke();}
        // Flowers bloom in summer; fruit replaces them as autumn arrives.
        if(summer>.002&&i%11===0) {
          ctx.fillStyle=rgb(p.flower,summer*.95);ctx.beginPath();
          for(let petal=0;petal<5;petal++){const a=petal*tau/5;ctx.ellipse(Math.cos(a)*leaf.r*.48,Math.sin(a)*leaf.r*.48,leaf.r*.5,leaf.r*.35,a,0,tau);}ctx.fill();
          ctx.fillStyle=`rgba(228,185,98,${summer})`;ctx.beginPath();ctx.arc(0,0,leaf.r*.2,0,tau);ctx.fill();
        }
        ctx.restore();
      });
      if(autumn>.002)for(const fruit of tree.fruits) {
        const x=bend(tree,fruit.x,fruit.y)+Math.sin(time*.5+fruit.phase)*wind*1.2,r=fruit.r*(.35+.65*Math.sqrt(autumn));
        ctx.strokeStyle=`rgba(129,105,64,${autumn*.9})`;ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(x,fruit.y-r*1.8);ctx.lineTo(x,fruit.y);ctx.stroke();
        ctx.fillStyle=`rgba(206,111,58,${autumn*.95})`;ctx.beginPath();ctx.arc(x,fruit.y,r,0,tau);ctx.fill();
        ctx.fillStyle=`rgba(250,204,139,${autumn*.65})`;ctx.beginPath();ctx.ellipse(x-r*.25,fruit.y-r*.3,r*.23,r*.16,-.6,0,tau);ctx.fill();
      }
    }
    // Rain arrives with summer clouds and fades as autumn winds strengthen.
    if(summer>.002) {
      for(let layer=0;layer<3;layer++) {
        ctx.strokeStyle=`rgba(191,213,226,${summer*(.12+layer*.045)})`;ctx.lineWidth=.55+layer*.22;ctx.beginPath();
        for(let i=0;i<110;i++) {
          const n=i+layer*110,travel=(time*(.75+layer*.17)+n*.618)%1,y=travel*height,x=((n*.754877%1)*width+travel*(18+wind*28))%width,len=9+layer*7;
          ctx.moveTo(x,y);ctx.lineTo(x-3-layer*2,y+len);
        }ctx.stroke();
      }
      for(let i=0;i<25;i++) {
        const travel=(time*.6+i*.723)%1,x=width*((i*.619)%1),y=height*(.71+(i%6)*.025);
        ctx.strokeStyle=rgb(p.accent,summer*(1-travel)*.16);ctx.lineWidth=.65;ctx.beginPath();ctx.ellipse(x,y,2+travel*12,.5+travel*2.1,0,0,tau);ctx.stroke();
      }
    }
    if(autumn>.002) {
      for(let i=0;i<64;i++) {
        const tree=trees[i%2],leaf=tree.leaves[(i*37)%tree.leaves.length],travel=(time*(.09+(i%4)*.008)+i*.381)%1;
        const x=leaf.x+travel*width*.19+Math.sin(time*.7+i)*18,y=leaf.y+travel*(height-leaf.y)*1.03;
        ctx.save();ctx.translate(x,y);ctx.rotate(time*(.7+(i%3)*.2)+i);ctx.fillStyle=`rgba(${i%3===0?'177,101,55':i%3===1?'219,167,85':'188,135,64'},${autumn*.8*Math.sin(travel*Math.PI)})`;
        ctx.beginPath();ctx.ellipse(0,0,3.5+(i%3),1.8,0,0,tau);ctx.fill();ctx.restore();
      }
      for(let i=0;i<7;i++) {
        const x=width*((i*.273+time*.009)%1),y=height*(.22+(i%4)*.18);
        ctx.strokeStyle=rgb(p.accent,autumn*.065);ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(x,y);ctx.bezierCurveTo(x+30,y-8,x+65,y+7,x+105,y-3);ctx.stroke();
      }
    }
    if(winter>.002) {
      for(let i=0;i<160;i++) {
        const travel=(time*(.022+(i%5)*.004)+i*.618)%1,y=travel*height,x=((i*.754877%1)*width+Math.sin(time*.27+i)*20+travel*35)%width,r=1+(i%4)*.55;
        ctx.fillStyle=rgb(p.snow,winter*(.32+(i%4)*.1)*Math.pow(Math.sin(travel*Math.PI),.4));ctx.beginPath();ctx.arc(x,y,r,0,tau);ctx.fill();
        if(i%13===0){ctx.strokeStyle=rgb(p.snow,winter*.43);ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(x-r*1.8,y);ctx.lineTo(x+r*1.8,y);ctx.moveTo(x,y-r*1.8);ctx.lineTo(x,y+r*1.8);ctx.stroke();}
      }
    }
    const vignette=ctx.createRadialGradient(width*.5,height*.43,width*.20,width*.5,height*.48,width*.75);
    vignette.addColorStop(0,'#08141200');vignette.addColorStop(1,'#08141277');ctx.fillStyle=vignette;ctx.fillRect(0,0,width,height);
  }
  return {resize,paint,cycle};
};
