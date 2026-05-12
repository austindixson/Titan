import React from 'react';
import {AbsoluteFill, Composition, useCurrentFrame, interpolate} from 'remotion';
const Card = ({title, subtitle, mode}: {title:string; subtitle:string; mode:'install'|'setup'}) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 20], [0, 1], {extrapolateRight: 'clamp'});
  return <AbsoluteFill style={{background:'radial-gradient(circle at 20% 20%, #1f2937 0%, #0b1020 45%, #05070f 100%)',color:'white',fontFamily:'Inter, system-ui',padding:70,opacity:o}}>
    <div style={{fontSize:26,opacity:0.9,letterSpacing:1}}>GitHub Quickstart</div>
    <div style={{fontSize:68,fontWeight:800,marginTop:18,lineHeight:1.04}}>{title}</div>
    <div style={{fontSize:30,marginTop:14,color:'#a5b4fc'}}>{subtitle}</div>
    <div style={{marginTop:52,border:'1px solid #334155',borderRadius:16,padding:24,background:'rgba(15,23,42,0.55)'}}>
      {mode==='install' ? <><div style={{fontSize:24,color:'#93c5fd',marginBottom:12}}>Install / Clone</div><code style={{fontSize:27,color:'#e2e8f0'}}>git clone https://github.com/austindixson/Titan.git</code></> : <><div style={{fontSize:24,color:'#86efac',marginBottom:12}}>Setup / Run</div><code style={{fontSize:25,color:'#e2e8f0'}}># see README for project-specific commands</code></>}
    </div>
    <div style={{position:'absolute',right:52,bottom:40,fontSize:20,color:'#94a3b8'}}>Generated with Remotion</div>
  </AbsoluteFill>;
};
export const RemotionRoot = () => <>
  <Composition id="InstallFlow" component={() => <Card title="Titan" subtitle="Resilient local-first Titan TUI agent harness" mode="install" />} durationInFrames={60} fps={30} width={1600} height={900} />
  <Composition id="SetupFlow" component={() => <Card title="Titan" subtitle="Resilient local-first Titan TUI agent harness" mode="setup" />} durationInFrames={60} fps={30} width={1600} height={900} />
</>;
