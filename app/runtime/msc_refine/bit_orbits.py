"""Exact15 nucleotide-label orbit counts via partition-lattice inversion.

Two state bitplanes and one valid plane encode32sites/word. Pair/triple/double-
pair equality events and integer Mobius inversion give the15 histogram bins.
No nucleotide-model assumption or site/locus sampling is introduced.
"""
from __future__ import annotations
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def numpy_encode(states):
    x=np.asarray(states)
    if x.ndim!=2:raise ValueError('taxa x sites states required')
    width=(x.shape[1]+31)//32*32;padded=np.full((len(x),width),4,dtype=np.uint8);padded[:,:x.shape[1]]=x
    valid=padded<4;out=[]
    for value in [valid&((padded&1)!=0),valid&((padded&2)!=0),valid]:
        out.append(np.packbits(value,axis=1,bitorder='little').copy().view('<u4'))
    return np.stack(out,axis=1)


def numpy_counts(planes,queries,regions):
    """Independent CPU equality-mask reference with Python int.bit_count."""
    result=np.zeros((len(queries),len(regions),15),dtype=np.int64)
    for q,(a,b,c,d) in enumerate(queries):
        for gene,(start,end) in enumerate(regions):
            start,end=int(start),int(end)
            for word in range(start//32,(end+31)//32):
                mask=((1<<min(32,end-32*word))-1)^((1<<max(0,start-32*word))-1)
                av,bv,cv,dv=[tuple(map(int,planes[i,:,word])) for i in [a,b,c,d]]
                valid=mask&av[2]&bv[2]&cv[2]&dv[2]
                pair=[]
                for x,y in [(av,bv),(av,cv),(av,dv),(bv,cv),(bv,dv),(cv,dv)]:
                    pair.append(~((x[0]^y[0])|(x[1]^y[1])))
                ab,ac,ad,bc,bd,cd=pair;pc=int.bit_count
                same=pc(valid&ab&ac&ad)
                abc=pc(valid&ab&ac)-same;abd=pc(valid&ab&ad)-same
                acd=pc(valid&ac&ad)-same;bcd=pc(valid&bc&bd)-same
                abcd=pc(valid&ab&cd)-same;acbd=pc(valid&ac&bd)-same;adbc=pc(valid&ad&bc)-same
                sab=pc(valid&ab)-same-abc-abd-abcd;sac=pc(valid&ac)-same-abc-acd-acbd
                sad=pc(valid&ad)-same-abd-acd-adbc;sbc=pc(valid&bc)-same-abc-bcd-adbc
                sbd=pc(valid&bd)-same-abd-bcd-acbd;scd=pc(valid&cd)-same-acd-bcd-abcd
                different=pc(valid)-same-abc-abd-acd-bcd-abcd-acbd-adbc-sab-sac-sad-sbc-sbd-scd
                result[q,gene]+=np.array([same,abc,abd,abcd,sab,acd,acbd,sac,adbc,bcd,sbc,sad,sbd,scd,different])
    return result


@triton.jit
def _encode(P,O,STRIDE,L,W:tl.constexpr,B:tl.constexpr,FLAT:tl.constexpr=False):
    if FLAT:
        program=tl.program_id(0).to(tl.int64)
        taxon=program//tl.cdiv(W,B);block=program%tl.cdiv(W,B)
    else:
        taxon=tl.program_id(0).to(tl.int64);block=tl.program_id(1)
    word=block*B+tl.arange(0,B)
    bit=tl.arange(0,32);site=word[:,None]*32+bit[None,:]
    value=tl.load(P+taxon*STRIDE+site//2,site<L,other=0x44).to(tl.int32)
    value=(value>>((site%2)*4))&15;valid=(site<L)&(value<4)
    shift=1<<bit[None,:]
    low=tl.sum(tl.where(valid&((value&1)!=0),shift,0),1).to(tl.uint32)
    high=tl.sum(tl.where(valid&((value&2)!=0),shift,0),1).to(tl.uint32)
    observed=tl.sum(tl.where(valid,shift,0),1).to(tl.uint32)
    tl.store(O+(taxon*3+0)*W+word,low,word<W)
    tl.store(O+(taxon*3+1)*W+word,high,word<W)
    tl.store(O+(taxon*3+2)*W+word,observed,word<W)


@triton.jit
def _events(P,Q,R,OUT,W:tl.constexpr,G:tl.constexpr,TILES:tl.constexpr,B:tl.constexpr,
            TILE_OFFSET=0,FLAT:tl.constexpr=False):
    if FLAT:
        program=tl.program_id(0).to(tl.int64)
        block=program%TILES;gene=program//TILES%G;q=program//(TILES*G)
    else:
        q=tl.program_id(0).to(tl.int64);gene=tl.program_id(1);block=tl.program_id(2)
    start=tl.load(R+gene*2);end=tl.load(R+gene*2+1)
    word=start//32+(block+TILE_OFFSET)*B+tl.arange(0,B);inside=(word<(end+31)//32)&(word<W)
    a=tl.load(Q+q*4);b=tl.load(Q+q*4+1);c=tl.load(Q+q*4+2);d=tl.load(Q+q*4+3)
    a0=tl.load(P+(a*3)*W+word,inside,other=0);a1=tl.load(P+(a*3+1)*W+word,inside,other=0);av=tl.load(P+(a*3+2)*W+word,inside,other=0)
    b0=tl.load(P+(b*3)*W+word,inside,other=0);b1=tl.load(P+(b*3+1)*W+word,inside,other=0);bv=tl.load(P+(b*3+2)*W+word,inside,other=0)
    c0=tl.load(P+(c*3)*W+word,inside,other=0);c1=tl.load(P+(c*3+1)*W+word,inside,other=0);cv=tl.load(P+(c*3+2)*W+word,inside,other=0)
    d0=tl.load(P+(d*3)*W+word,inside,other=0);d1=tl.load(P+(d*3+1)*W+word,inside,other=0);dv=tl.load(P+(d*3+2)*W+word,inside,other=0)
    lo=tl.maximum(0,tl.minimum(32,start-word*32));hi=tl.maximum(0,tl.minimum(32,end-word*32))
    full=tl.full((B,),0xffffffff,tl.uint32)
    left=tl.where(lo>=32,0,full<<tl.minimum(lo,31));right=tl.where(hi<=0,0,full>>tl.minimum(32-hi,31))
    valid=av&bv&cv&dv&left&right
    ab=~((a0^b0)|(a1^b1));ac=~((a0^c0)|(a1^c1));ad=~((a0^d0)|(a1^d1))
    bc=~((b0^c0)|(b1^c1));bd=~((b0^d0)|(b1^d1));cd=~((c0^d0)|(c1^d1))
    same=libdevice.popc(valid&ab&ac&ad)
    abc=libdevice.popc(valid&ab&ac)-same;abd=libdevice.popc(valid&ab&ad)-same
    acd=libdevice.popc(valid&ac&ad)-same;bcd=libdevice.popc(valid&bc&bd)-same
    abcd=libdevice.popc(valid&ab&cd)-same;acbd=libdevice.popc(valid&ac&bd)-same;adbc=libdevice.popc(valid&ad&bc)-same
    sab=libdevice.popc(valid&ab)-same-abc-abd-abcd;sac=libdevice.popc(valid&ac)-same-abc-acd-acbd
    sad=libdevice.popc(valid&ad)-same-abd-acd-adbc;sbc=libdevice.popc(valid&bc)-same-abc-bcd-adbc
    sbd=libdevice.popc(valid&bd)-same-abd-bcd-acbd;scd=libdevice.popc(valid&cd)-same-acd-bcd-abcd
    different=libdevice.popc(valid)-same-abc-abd-acd-bcd-abcd-acbd-adbc-sab-sac-sad-sbc-sbd-scd
    base=OUT+((q*G+gene)*TILES+block)*15
    tl.store(base+0,tl.sum(same,0));tl.store(base+1,tl.sum(abc,0))
    tl.store(base+2,tl.sum(abd,0));tl.store(base+3,tl.sum(abcd,0))
    tl.store(base+4,tl.sum(sab,0));tl.store(base+5,tl.sum(acd,0))
    tl.store(base+6,tl.sum(acbd,0));tl.store(base+7,tl.sum(sac,0))
    tl.store(base+8,tl.sum(adbc,0));tl.store(base+9,tl.sum(bcd,0))
    tl.store(base+10,tl.sum(sbc,0));tl.store(base+11,tl.sum(sad,0))
    tl.store(base+12,tl.sum(sbd,0));tl.store(base+13,tl.sum(scd,0))
    tl.store(base+14,tl.sum(different,0))


@torch.inference_mode()
def encode(packed,length):
    if not packed.is_cuda or packed.dtype!=torch.uint8 or packed.ndim!=2 or packed.shape[1]!=(length+1)//2:
        raise ValueError('resident native packed uint8 MSA required')
    words=triton.cdiv(length,32);out=torch.empty((len(packed),3,words),device=packed.device,dtype=torch.uint32)
    blocks=triton.cdiv(words,64);flat=blocks>65535
    grid=(len(packed)*blocks,) if flat else (len(packed),blocks)
    _encode[grid](packed,out,packed.shape[1],length,words,64,flat,num_warps=4)
    return out


@torch.inference_mode()
def counts(planes,queries,regions,*,word_block=128):
    if not planes.is_cuda or planes.dtype!=torch.uint32 or planes.ndim!=3 or planes.shape[1]!=3:raise ValueError('resident3-bitplane array required')
    q=torch.as_tensor(queries,device=planes.device,dtype=torch.int64).contiguous()
    rr=np.asarray(regions,dtype=np.int64)
    if q.ndim!=2 or q.shape[1]!=4 or rr.ndim!=2 or rr.shape[1]!=2 or len(rr)==0:raise ValueError('Qx4 andGx2 intervals required')
    if (rr[:,0]<0).any() or (rr[:,1]<=rr[:,0]).any() or (rr[:,1]>planes.shape[2]*32).any():raise ValueError('invalid intervals')
    if bool(((q<0)|(q>=len(planes))).any()) or bool((q.sort(1).values.diff(dim=1)==0).any()):raise ValueError('four distinct in-range query taxa required')
    region=torch.as_tensor(rr,device=planes.device)
    tiles=int(triton.cdiv(int(((rr[:,1]+31)//32-rr[:,0]//32).max()),word_block))
    # Execution caller batches queries/loci/sites forbounded scratch.
    out=torch.empty((len(q),len(rr),tiles,15),device=planes.device,dtype=torch.int32)
    if len(q):
        flat=len(rr)>65535 or tiles>65535
        grid=(len(q)*len(rr)*tiles,) if flat else (len(q),len(rr),tiles)
        _events[grid](planes,q,region,out,planes.shape[2],len(rr),tiles,word_block,FLAT=flat,num_warps=4)
    return out.sum(2,dtype=torch.int64)
