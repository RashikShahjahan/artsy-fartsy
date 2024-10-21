import { Canvas } from '@react-three/fiber'
import Line from './Line';
import './App.css';
import { useState } from 'react';
/*Tasks:
Setup react-three/fiber with react 
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
  };
}

function App() {
  const [input, setInput] = useState('');
  const [drawCommands, setDrawCommands] = useState<DrawCommand[]>([]);
  
  function submitInput() {
    console.log(input);
    setDrawCommands([
      { type: 'line', args: { start: [0, 0, 0], end: [1, 1, 1] } },
      { type: 'line', args: { start: [0, 2, 0], end: [4, 1, 1] } },
    ]);
  }

  return (
    <div>
      <Canvas>
        {drawCommands.map((command, index) => (
          <Line key={index} start={command.args.start} end={command.args.end} />
        ))}
      </Canvas>
      <input type="text" value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={() => submitInput()}>Draw</button>
    </div>
  )
}

export default App;