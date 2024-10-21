import { useLayoutEffect, useRef } from "react";
import * as THREE from 'three';

function Line({ start, end }: { start: number[], end: number[] }) {
    const ref = useRef<THREE.BufferGeometry>(null);
    useLayoutEffect(() => {
      if (ref.current) {
        ref.current.setFromPoints([start, end].map((point) => new THREE.Vector3(...point)));
      }
    }, [start, end]);
    return (
      <line>
        <bufferGeometry ref={ref} />
        <lineBasicMaterial color="hotpink" />
      </line>
    )
  }

export default Line;