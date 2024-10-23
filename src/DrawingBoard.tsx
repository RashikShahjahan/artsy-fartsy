/*Tasks:
Saving Drawings and viewing gallery(Add auth)
  - View to Display drawings in their feed in a news feed(Make it like a gallery)
  - GET request to retrieve_gallery
  - Post request to like/unlike drawings

Styling:


Enhancements:
  - User profile page
  - View to display user's saved drawings
  - View to display user's followers and following
  - Help button to help artists with code syntax
  - Call LLM to generate code
*/
import { useState} from 'react';
import { useAuth } from '@clerk/clerk-react';
import { saveDrawingToServer, submitDrawingCommands } from './api';
import { Command } from './types';
import { Canvas } from '@react-three/fiber';
import Line from './Line';
import Arc from './Arc';
import SignInButtonWrapper from './components/SignInButtonWrapper';

function DrawingBoard({auth}: {auth: boolean}) {
  const [input, setInput] = useState('');
  const [drawCommands, setDrawCommands] = useState<Command[]>([]);
  const { getToken } = useAuth();

  const handleSubmitInput = async () => {
    try {
      const token = await getToken();
      if (!token) {
        throw new Error('No token found');
      }
      const commands = await submitDrawingCommands(input, token);
      setDrawCommands(commands);
    } catch (error) {
      console.error('Error interpreting code:', error);
    }
  };

  async function handleSaveDrawing() {
    try {
      const token = await getToken() ?? '';

      await saveDrawingToServer(drawCommands, token);
      console.log('Drawing saved successfully');
    } catch (error) {
      console.error('Error saving drawing:', error);
      // You might want to show an error message to the user here
    }
  }

  return (
    <div className="flex flex-row items-start justify-center min-h-screen bg-gray-100 p-8 gap-8">
      <div className="w-1/2 h-[calc(100vh-4rem)] bg-white rounded-lg shadow-md p-6 flex flex-col relative">
        <Canvas>
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
        {auth ? (
          <button 
            className="absolute bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black hover:bg-yellow-350 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50"
            onClick={handleSaveDrawing}
          >
            Save
          </button>
        ) : (
          <SignInButtonWrapper />
        )}
      </div>
      <div className="w-1/2 h-[calc(100vh-4rem)] bg-white rounded-lg shadow-md p-6 flex flex-col">
        <textarea 
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          placeholder="Enter drawing commands"
          className="w-full flex-grow border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button 
          onClick={handleSubmitInput} 
          className="w-full px-6 py-3 bg-pink-400 text-black hover:bg-pink-350 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-pink-200 focus:ring-opacity-50 mt-6"
        >
          Draw
        </button>
      </div>
    </div>
  );
}

export default DrawingBoard;
