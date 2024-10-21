import { useLayoutEffect, useRef } from "react";
import * as THREE from 'three';

function Arc({ start, end, center, startAngle, endAngle, clockwise, rotation }: { 
  start: number[], 
  end: number[], 
  center: number[], 
  startAngle: number, 
  endAngle: number, 
  clockwise: boolean, 
  rotation: number 
}) {
    const ref = useRef<THREE.BufferGeometry>(null);
    useLayoutEffect(() => {
      if (ref.current) {
        const curve = new THREE.EllipseCurve(
          center[0], center[1],                // Center x, y
          Math.hypot(start[0] - center[0], start[1] - center[1]), // xRadius
          Math.hypot(start[0] - center[0], start[1] - center[1]), // yRadius
          startAngle, endAngle,                // Start angle, end angle
          clockwise,                           // Clockwise
          rotation                             // Rotation
        );
        const points = curve.getPoints(50);
        ref.current.setFromPoints(points);
      }
    }, [start, end, center, startAngle, endAngle, clockwise, rotation]);
    return (
      <line>
        <bufferGeometry ref={ref} />
        <lineBasicMaterial color="#FFD700" />
      </line>
    )
  }

export default Arc;