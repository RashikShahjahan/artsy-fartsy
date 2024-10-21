import { useLayoutEffect, useRef } from "react";
import * as THREE from 'three';

function Line({ start, end, color }: { start: number[], end: number[], color: string }) {
    const ref = useRef<THREE.BufferGeometry>(null);
    useLayoutEffect(() => {
      if (ref.current) {
        ref.current.setFromPoints([start, end].map((point) => new THREE.Vector3(...point)));
        console.log(start, end);
      }
    }, [start, end]);
    return (
      <line>
        <bufferGeometry ref={ref} />
        <lineBasicMaterial color={color} />
      </line>
    )
  }

export default Line;