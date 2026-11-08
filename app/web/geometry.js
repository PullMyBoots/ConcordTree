'use strict';
// Original growing geometric life trees, retained as an independent theme.
window.growingLandscape = canvas => {
  const ctx=canvas.getContext('2d');
  const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
  let width=0,height=0,branches=[],elapsed=2800;
  const duration=23000;
  // Small original line illustrations share one 32-unit visual language.
  const species = {
    bird: ['M3 23c5 0 8-3 10-8L8 7c5 0 9 3 11 6l4-3 4 1 3 3-5 1c-1 8-7 12-15 10L3 28l3-5', 'M24 12h.1M14 18l6 2'],
    butterfly: ['M16 14C7-3 0 6 5 16c-5 8 4 14 11 3 7 11 16 5 11-3 5-10-2-19-11-2Z', 'M16 11v14M16 11l-3-5m3 5 3-5'],
    fish: ['M5 16c5-10 15-10 22 0-7 10-17 10-22 0Zm0 0-4-6v12l4-6', 'M20 10c-3 3-3 9 0 12M23 15h.1M12 9l3-4 4 3M12 23l4 4 3-4'],
    leaf: ['M7 26C-1 10 16 2 28 4c0 14-5 26-18 20', 'M5 29 23 9M11 22l1-9m5 3 8-1'],
    fern: ['M12 30 20 3M17 14C3 15 4 5 17 10M19 9c1-7 9-7 8-2l-8 5M15 21C1 22 1 12 15 17M17 18c9-12 17-7 8-1l-8 5M13 27C2 30 0 22 13 23M15 25c11-11 17-4 7 0l-8 4'],
    flower: ['M16 14c-8-14 9-17 5-3 14-4 13 12 0 9 7 13-9 18-9 3C-1 31-5 15 9 16-4 6 10-3 13 11Z', 'M16 13a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z'],
    mushroom: ['M3 18C3 2 29 2 29 18H3Z', 'M13 18l-2 11h10l-2-11M9 12h.1M17 9h.1M23 13h.1'],
    turtle: ['M7 19c-1-14 19-17 19-1L7 19ZM26 16c8-5 8 5 0 4M9 19l-4 5 5-1 3-4M21 19l4 5-5-1-3-4M7 17l-4 1', 'M11 10l5 3 7-3M16 13v6'],
    deer: ['M10 29V19l-5-5 4-1 4 3h7l4-3 4 1-6 5v10M13 23l3 4 3-4', 'M11 16 8 8 4 7 3 3M8 8l3-4M21 16l3-8 4-1 1-4M24 8l-3-4M12 20h.1M20 20h.1'],
    dragonfly: ['M16 7v23M14 5a2 2 0 1 0 4 0 2 2 0 0 0-4 0M16 13C-2-1-3 16 16 16c19 0 18-17 0-3ZM16 17C0 12 0 26 16 21c16 5 16-9 0-4Z'],
    jellyfish: ['M5 17C3-2 29-2 27 17c-3-3-4 3-7 0-3-3-5 3-8 0-3-3-4 3-7 0ZM9 18c-6 8 7 6 1 13M16 18c-5 8 5 7 0 13M23 18c-6 8 7 6 1 13'],
    snail: ['M13 23C-2 23 1 4 13 5c14 1 11 19 0 17-7-1-6-11 0-11 7 0 5 8 0 6', 'M4 23v4h22c4-2 4-8 0-9l-5 5H4M26 18l-2-5M28 18l3-4']
  };
  const icons = Object.fromEntries(Object.entries(species).map(([name,paths]) => [name, paths.map(path=>new Path2D(path))]));
  function buildTrees() {
    branches = [];
    const mobile = width < 700;
    const layouts = mobile ? [
      [[.11,.09,'bird'],[.37,.14,'butterfly'],[.65,.075,'leaf'],[.88,.15,'fish']]
    ] : [
      [[.15,.13,'bird'],[.052,.235,'butterfly'],[.25,.29,'dragonfly'],[.105,.39,'leaf'],[.265,.48,'fern'],[.045,.57,'mushroom'],[.20,.66,'flower'],[.095,.75,'snail']],
      [[.825,.15,'butterfly'],[.95,.245,'bird'],[.745,.32,'deer'],[.89,.415,'turtle'],[.755,.52,'fish'],[.955,.60,'jellyfish'],[.805,.68,'leaf'],[.905,.77,'fern']]
    ];
    for (let side=0;side<layouts.length;side++) {
      const leaves = layouts[side].map(([x,y,icon])=>({x:x*width,y:y*height,icon}));
      function node(items) {
        if(items.length===1)return items[0];
        const mid=Math.floor(items.length/2), children=[node(items.slice(0,mid)),node(items.slice(mid))];
        return {x:(children[0].x+children[1].x)/2+(side ? -1 : 1)*width*.009,
          y:Math.max(...children.map(child=>child.y))+height*(mobile ? .065 : .057),children};
      }
      let crown;
      if(mobile) crown=node(leaves);
      else {
        crown=leaves[0];
        for(let i=1;i<leaves.length;i++) {
          crown={x:width*((side ? .84 : .16)+Math.sin(i*1.7+side)*.012),
            y:leaves[i].y+height*.055,children:[crown,leaves[i]]};
        }
      }
      const root={x:crown.x+(side ? -.035 : .035)*width,y:mobile ? height*.47 : height*1.03};
      function connect(start,end,depth,order) {
        const c1={x:start.x+(end.x-start.x)*.12,y:start.y+(end.y-start.y)*.58};
        const c2={x:end.x-(end.x-start.x)*.22,y:end.y-(end.y-start.y)*.12};
        const points=[];
        for(let i=0;i<=32;i++){
          const t=i/32,u=1-t;
          points.push({x:u*u*u*start.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*end.x,
            y:u*u*u*start.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*end.y});
        }
        branches.push({points,end,depth,side,delay:depth*1150+order*100+side*430});
        end.children?.forEach((child,index)=>connect(end,child,depth+1,order+index));
      }
      connect(root,crown,0,0);
    }
  }
  function paint(milliseconds=elapsed) {
    elapsed=milliseconds;
    if (!ctx) return;
    ctx.clearRect(0,0,width,height);
    const still=reducedMotion.matches || Boolean(navigator.connection?.saveData);
    const time=still ? 16000 : elapsed % duration;
    const fade=time>19500 ? Math.max(0,1-(time-19500)/2800) : 1;
    ctx.lineCap='round';ctx.lineJoin='round';
    function stroke(points,amount,sway) {
      const step=amount*(points.length-1),last=Math.floor(step);
      ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);
      for(let i=1;i<=last;i++)ctx.lineTo(points[i].x+sway*i/32,points[i].y);
      if(last<points.length-1){const a=points[last],b=points[last+1],part=step-last;
        ctx.lineTo(a.x+(b.x-a.x)*part+sway*step/32,a.y+(b.y-a.y)*part);}
      ctx.stroke();
    }
    for(const branch of branches) {
      const {points,end,depth,side,delay}=branch;
      const raw=Math.max(0,Math.min(1,(time-delay)/1100));
      const progress=raw*raw*(3-2*raw);
      const sway=still ? 0 : Math.sin(time/3600+side+depth*.2)*1.4;
      const rgb=side ? '211,202,164' : '165,205,180';
      ctx.strokeStyle=`rgba(${rgb},.042)`;ctx.lineWidth=.8;stroke(points,1,sway);
      if(!progress || !fade)continue;
      ctx.strokeStyle=`rgba(${rgb},${Math.max(.18,.38-depth*.025)*fade})`;
      ctx.lineWidth=Math.max(.6,1.3-depth*.09);stroke(points,progress,sway);
      if(progress<1){
        const tip=points[Math.floor(progress*32)];
        ctx.fillStyle=`rgba(${rgb},${.5*fade})`;ctx.shadowColor=`rgba(${rgb},.4)`;ctx.shadowBlur=9;
        ctx.beginPath();ctx.arc(tip.x+sway*progress,tip.y,1.5,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
      }
      if(end.icon){
        const reveal=Math.max(0,Math.min(1,(time-delay-850)/1000));
        if(!reveal)continue;
        const size=(width<700 ? 27 : Math.max(28,Math.min(36,width*.025)));
        ctx.save();ctx.translate(end.x+sway,end.y);
        // A quiet ink-colored disc keeps the branch out of the illustration.
        ctx.globalAlpha=reveal*fade;ctx.fillStyle='#11221a';
        ctx.beginPath();ctx.arc(0,0,size*.64,0,Math.PI*2);ctx.fill();
        ctx.scale(size/32*(.85+.15*reveal),size/32*(.85+.15*reveal));ctx.translate(-16,-16);
        ctx.strokeStyle=side ? 'rgba(220,211,175,.75)' : 'rgba(183,216,193,.8)';ctx.lineWidth=1.05;
        for(const path of icons[end.icon])ctx.stroke(path);
        ctx.restore();
      } else if(progress===1){
        ctx.fillStyle=`rgba(${rgb},${.45*fade})`;
        ctx.beginPath();ctx.arc(end.x+sway,end.y,1.8,0,Math.PI*2);ctx.fill();
      }
    }
  }
  function resize() {
    width = innerWidth; height = innerHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    if (ctx) ctx.setTransform(ratio,0,0,ratio,0,0);
    buildTrees(); paint();
  }
  return {resize,paint};
};
