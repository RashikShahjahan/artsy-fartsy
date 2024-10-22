
/*Tasks:
Saving Drawings and viewing gallery(Add auth)
  - Create button to save drawings
  - Axios POST request to /save_art
  - View to Display drawings in their feed in a news feed(Make it like a gallery)
  - GET request to retrieve_gallery
  - Post request to like/unlike drawings

Styling:
  - Sign in button and sign out button should be styled
  - Everything else should be styled

Enhancements:
  - User profile page
  - View to display user's saved drawings
  - View to display user's followers and following
  - Help button to help artists with code syntax
  - Call LLM to generate code
*/
import { Canvas } from '@react-three/fiber'
import Line from './Line';
import Arc from './Arc';
import { useState } from 'react';
import axios from 'axios';
import { useAuth } from '@clerk/clerk-react';

type CommandArgs = (number | string | boolean)[];

type Command = {
    type: 'line' | 'arc';
    args: CommandArgs;
}

function DrawingBoard() {
  const [input, setInput] = useState('');
  const [drawCommands, setDrawCommands] = useState<Command[]>([]);
  const { getToken } = useAuth();
  async function submitInput() {
    try {
      const token = await getToken();
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
      const response = await axios.post(`${apiBaseUrl}/interpret`, { code: input }, {
        headers: {
           Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });
      if (response.status !== 200) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const commands = response.data;
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
        <div className="flex justify-between gap-4">
          <button 
            onClick={() => submitInput()} 
          className="w-1/2 px-6 py-3 bg-pink-400 text-black  hover:bg-pink-350 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-pink-200 focus:ring-opacity-50"
        >
            Draw
          </button>
          <button className="w-1/2 px-6 py-3 bg-yellow-400 text-black  hover:bg-yellow-350 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50">
            Save
          </button>
        </div>
      </div>
    </div>
  )
}

export default DrawingBoard;