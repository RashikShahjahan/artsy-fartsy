import { useState} from 'react';
import { useAuth } from '@clerk/clerk-react';
import { saveDrawingToServer, submitDrawingCommands, aiDrawingCommands } from './api';
import { Command } from './types';
import { Canvas } from '@react-three/fiber';
import Line from './Line';
import Arc from './Arc';
import SignInButtonWrapper from './components/SignInButtonWrapper';
import { SignInButton } from '@clerk/clerk-react';

function DrawingBoard({auth}: {auth: boolean}) {
  const [input, setInput] = useState('');
  const [aiMode, setAiMode] = useState(false);
  const [drawCommands, setDrawCommands] = useState<Command[]>([]);
  const { getToken } = useAuth();

  const handleSubmitInput = async () => {
      const token = await getToken();

      const commands = await submitDrawingCommands(input, token);
      setDrawCommands(commands);

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

  const handleGenerateCode = async () => {
    try {
      const token = await getToken();
      if (!token) {
        throw new Error('No token found');
      }
      const commands = await aiDrawingCommands(input, token);
      console.log(commands);
      setDrawCommands(commands);
    } catch (error) {
      console.error('Error generating code:', error);
      // You might want to show an error message to the user here
    }
  };

  const handleDrawClick = () => {
    if (aiMode) {
      handleGenerateCode();
    } else {
      handleSubmitInput();
    }
  };

  return (
    <div className="flex flex-col md:flex-row items-start justify-center gap-8">
      <div className="w-full md:w-1/2 h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col relative">
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
            className="absolute bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black rounded-lg hover:bg-yellow-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50"
            onClick={handleSaveDrawing}
          >
            Save
          </button>
        ) : (
          <SignInButtonWrapper text="Sign in to save your art!" />
        )}
      </div>
      <div className="w-full md:w-1/2 h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col">
        <div className="mb-4 flex items-center justify-between">
          <span className="text-lg font-semibold">
            {aiMode ? "AI Mode" : "Normal Mode"}
          </span>
          {auth ? (
            <button
              onClick={() => setAiMode(!aiMode)}
              className={`px-4 py-2 rounded-full transition-colors duration-300 ${
                aiMode
                  ? "bg-blue-500 text-white"
                  : "bg-gray-200 text-gray-800"
              }`}
            >
              {aiMode ? "Switch to Normal" : "Switch to AI"}
            </button>
          ) : (
            <SignInButton mode="modal">
              <button
                className="px-4 py-2 rounded-full transition-colors duration-300 bg-gray-200 text-gray-800"
              >
                Sign in to use AI
              </button>
            </SignInButton>
          )}
        </div>
        <textarea 
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          placeholder={aiMode ? "Describe your drawing" : "Enter drawing commands"}
          className="w-full flex-grow border border-gray-300 rounded-lg resize-none p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <div className="flex gap-4 mt-6">
          <button 
            onClick={handleDrawClick}
            className={`w-full px-6 py-3 text-black rounded-lg transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-opacity-50 ${
              aiMode
                ? "bg-blue-400 hover:bg-blue-300 focus:ring-blue-200"
                : "bg-pink-400 hover:bg-pink-300 focus:ring-pink-200"
            }`}
          >
            { "Draw"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default DrawingBoard;
