import { Canvas } from '@react-three/fiber'
import Line from './Line';
import Arc from './Arc';
import { useState } from 'react';
import { Command, interpret } from './interpret';
/*Tasks:
Setup react-three/fiber with react [Done]
    -Create component for ARC and LINE
    -Add textbox in react to call interpreter on button click
    -Create render function to render the output list of objects

Write version of interpreter to run single command->Line and arc:[DONE]
    -Create lexer for ARC and LINE
    -Parse tokens to create ARC and LINE objects
    -return ARC and LINE objects

Fix tailwind[Done]

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
    const commands = interpret(input);
    console.log(commands)
    setDrawCommands(commands);
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-[90%] h-96 bg-white rounded-lg shadow-md p-6 flex-grow">
        <Canvas className="w-full h-full mb-6 border border-gray-300 rounded">
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
                clockwise={command.args[8] as boolean}
                rotation={command.args[9]}
              />
          ))}
        </Canvas>
      </div>
      <div className="w-full max-w-[90%] mt-4">
        <textarea 
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          placeholder="Enter drawing commands"
          className="w-full mb-3 p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={10}
        />
        <button 
          onClick={() => submitInput()} 
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
        >
          Draw
        </button>
      </div>
    </div>)}

export default App;