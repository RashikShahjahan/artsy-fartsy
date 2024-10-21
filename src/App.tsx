import { Canvas } from '@react-three/fiber'
import Line from './Line';
import Arc from './Arc';
import './App.css';
import { useState } from 'react';
/*Tasks:
Setup react-three/fiber with react [Done]
    -Create component for ARC and LINE
    -Add textbox in react to call interpreter on button click
    -Create render function to render the output list of objects

Write version of interpreter to run single command->Line and arc:
    -Create lexer for ARC and LINE
    -Parse tokens to create ARC and LINE objects
    -return ARC and LINE objects

Enhancements:
    Add variables and expressions
    Add loops
    Add error validation
    Add coloring
*/
type DrawCommand = {
  type: 'line' | 'arc';
  args: {
    start: [number, number, number];
    end: [number, number, number];
    center?: [number, number, number];
    startAngle?: number;
    endAngle?: number;
    clockwise?: boolean;
    rotation?: number;
  };
}

function App() {
  const [input, setInput] = useState('');
  const [drawCommands, setDrawCommands] = useState<DrawCommand[]>([]);
  
  function submitInput() {
    console.log(input);
    setDrawCommands([
      { type: 'line', args: { start: [0, 0, 0], end: [1, 1, 1] } },
      { type: 'arc', args: { start: [0, 2, 0], end: [4, 1, 1], center: [2, 1.5, 0.5], startAngle: 0, endAngle: Math.PI, clockwise: true, rotation: 0 } },
    ]);
  }

  return (
    <div>
      <Canvas>
        {drawCommands.map((command, index) => (
          command.type === 'line' ? 
            <Line key={index} start={command.args.start} end={command.args.end} /> :
            <Arc 
              key={index} 
              start={command.args.start} 
              end={command.args.end} 
              center={command.args.center!} 
              startAngle={command.args.startAngle!} 
              endAngle={command.args.endAngle!} 
              clockwise={command.args.clockwise!} 
              rotation={command.args.rotation!} 
            />
        ))}
      </Canvas>
      <input type="text" value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={() => submitInput()}>Draw</button>
    </div>
  )
}

export default App;