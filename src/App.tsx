import { Canvas } from '@react-three/fiber'
import Line from './Line';
import Arc from './Arc';
import { useState } from 'react';
/*Tasks:

Create Express server to host interpreter
    -Create API endpoint to run interpreter
    -Use API endpoint in frontend

Saving Drawings
    -Create button to save drawings
    -Save drawings to Database
    -Load drawings from Database

Routing and Authentication
    -Create login/register
    -Create protected routes
    -Create logout

News Feed which lets you see all drawings

Enhancements:
    Add coloring[Done]
    Add variables and expressions
    Add loops
    Add error validation

*/

type CommandArgs = (number | string | boolean)[];

type Command = {
    type: 'line' | 'arc';
    args: CommandArgs;
}



function App() {
  const [input, setInput] = useState('');
  const [drawCommands, setDrawCommands] = useState<Command[]>([]);
  
  async function submitInput() {
    try {
      const apiBaseUrl = process.env.REACT_APP_API_BASE_URL;
      const response = await fetch(`${apiBaseUrl}/interpret`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code: input }),
      });
      const commands = await response.json();
      console.log(commands);
      setDrawCommands(commands);
    } catch (error) {
      console.error('Error interpreting code:', error);
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-[90%] h-96 bg-white rounded-lg shadow-md p-6 flex-grow">
        <Canvas className="w-full h-full mb-6 border border-gray-300 rounded">
          {drawCommands.map((command, index) => (
            command.type === 'line' ? 
              <Line key={index} start={[Number(command.args[0]), Number(command.args[1]), 0]} end={[Number(command.args[2]), Number(command.args[3]), 0]} color={command.args[4].toString()} /> :
              <Arc 
                key={index} 
                center={[Number(command.args[0]), Number(command.args[1]), 0]}
                start={[Number(command.args[2]), Number(command.args[3]), 0]}
                end={[Number(command.args[4]), Number(command.args[5]), 0]}
                startAngle={Number(command.args[6])}
                endAngle={Number(command.args[7])}
                clockwise={Boolean(command.args[8])}
                rotation={Number(command.args[9])}
                color={command.args[10].toString()}

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