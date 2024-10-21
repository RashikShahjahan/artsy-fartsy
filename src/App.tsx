import { Canvas } from '@react-three/fiber'
import Line from './Line';
import Arc from './Arc';
import './App.css';
import { useState } from 'react';
import { Command, interpret } from './interpret';
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


function App() {
  const [input, setInput] = useState('');
  const [drawCommands, setDrawCommands] = useState<Command[]>([]);
  
  function submitInput() {
    console.log(input);
    setDrawCommands(interpret(input));
  }

  return (
    <div className="flex w-full h-screen">
      <div className="w-1/2 h-full">
        <Canvas className="w-full h-full">
          {drawCommands.map((command, index) => (
            command.type === 'line' ? 
              <Line key={index} start={[command.args[0], command.args[1], 0]} end={[command.args[2], command.args[3], 0]} /> :
              <Arc 
                key={index} 
                center={[command.args[0], command.args[1], 0]}
                start={[command.args[2], command.args[3], 0]}
                end={[command.args[4], command.args[5], 0]}
                startAngle={command.args[6]}
                endAngle={command.args[7]}
                clockwise={command.args[11] as boolean}
                rotation={command.args[12]}
              />
          ))}
        </Canvas>
      </div>
      <div className="w-1/2 h-full flex flex-col justify-center items-center">
        <input 
          type="text" 
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          placeholder="Enter drawing commands"
          className="w-4/5 mb-4 p-2 border border-gray-300 rounded"
        />
        <button onClick={() => submitInput()} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
          Draw
        </button>
      </div>
    </div>
  )
}

export default App;